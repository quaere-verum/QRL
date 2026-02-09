import torch
from dataclasses import dataclass
from abc import abstractmethod
import copy as cp
from ..mdp import MDP
from ..functions import DeterministicPolicy, QuantileFunction, GaussianNoiseScale
from typing import Callable
from tqdm import tqdm
from typing import Literal
import math
SQRT_TWO = math.sqrt(2)
SQRT_TWO_PI = math.sqrt(2 / math.pi)
SQRT_TWOPI = math.sqrt(2 * math.pi)
   
def _standard_normal_cdf(
    x: torch.Tensor
) -> torch.Tensor:
    return (1.0 + torch.erf(x / SQRT_TWO)) / 2.0
 
def generalised_quantile_huber_loss(
    quantile_function: QuantileFunction,
    state: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    mse_bound_decay: float,
    *,
    approximate: bool = True,
) -> torch.Tensor:
    batch_size = target.shape[0]
    quantile_estimate = quantile_function.compute_quantiles(state, action)
 
    u = target[:, torch.newaxis, :] - quantile_estimate[:, :, torch.newaxis]
    u_abs = torch.abs(u)
    with torch.no_grad():
        b_new = torch.quantile(u_abs.detach(), 0.70)
        quantile_function.b.mul_(mse_bound_decay).add_((1 - mse_bound_decay) * b_new.detach())
 
    if approximate:
        loss = torch.where(
            u_abs < quantile_function.b,
            1.0 / (quantile_function.b * SQRT_TWOPI) * u ** 2,
            u_abs - quantile_function.b * SQRT_TWO_PI
        )
    else:
        loss = (
            u_abs * (1.0 - 2 * _standard_normal_cdf(-u_abs / quantile_function.b))
            + quantile_function.b * SQRT_TWO_PI * (torch.exp(-u ** 2 / (2 * quantile_function.b ** 2)) - 1.0)
        )
 
    multiplier = torch.abs(
        quantile_function.quantile_locations[torch.newaxis, :, torch.newaxis]
        - (u < 0).float()
    )
 
    return torch.sum(loss * multiplier) / (batch_size * quantile_function.nr_quantiles * quantile_function.nr_quantiles)
 
def quantile_huber_loss(
    quantile_function: QuantileFunction,
    state: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    mse_bound: float,
) -> torch.Tensor:
    batch_size = target.shape[0]
    quantile_estimate = quantile_function.compute_quantiles(state, action)
    u = target[:, torch.newaxis, :] - quantile_estimate[:, :, torch.newaxis]
    u_abs = torch.abs(u)
    huber_loss = torch.where(
        u_abs < mse_bound,
        0.5 * torch.pow(u, 2),
        mse_bound * (u_abs - 0.5 * mse_bound)
    )
    multiplier = torch.abs(
        quantile_function.quantile_locations[:, torch.newaxis]
        - (u.detach() < 0).float()
    )
    return torch.sum(huber_loss * multiplier) / (batch_size * quantile_function.nr_quantiles * quantile_function.nr_quantiles)
 
@dataclass
class D4PG_Base:
    mdp: MDP
    policy: DeterministicPolicy
    objective_function: Callable[[torch.Tensor], torch.Tensor]
    quantile_function: QuantileFunction
    policy_optimiser: torch.optim.Optimizer
    quantile_function_optimiser: torch.optim.Optimizer
    exploration_noise: GaussianNoiseScale
    episodes_per_buffer: int
    episodes_per_update: int
    batches_per_update: int
    gamma: float
    policy_update_delay: int
    rho_policy: float
    rho_critic: float
    batch_size: int
    normalise_rewards: bool = False
    policy_max_gradient_norm: float | None = None
    critic_max_gradient_norm: float | None = None
 
    def __post_init__(self):
        assert 0 < self.rho_policy < 1 and 0 < self.rho_critic < 1
        assert self.exploration_noise.initial_value.ndim == self.exploration_noise.minimum_value.ndim == 1
        assert self.exploration_noise.initial_value.shape[0] == self.mdp.action_dim
        assert self.exploration_noise.initial_value.shape == self.exploration_noise.minimum_value.shape
        assert self.policy_update_delay > 0 and self.batches_per_update % self.policy_update_delay == 0
 
        self._episode_ptr = 0
        self._nr_filled = 0
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_2_mean = 0.0
        self._state_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer, self.mdp.state_dim),
            dtype=torch.float32
        )
        self._action_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer, self.policy.action_dim),
            dtype=torch.float32
        )
        self._reward_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer),
            dtype=torch.float32
        )
        self._is_terminal = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer),
            dtype=torch.bool
        )
        self._is_terminal[:, -1, :] = True
 
        self._target_quantile_function = cp.deepcopy(self.quantile_function)
        self._target_policy = cp.deepcopy(self.policy)
        self._target_quantile_function.disable_training()
        self._target_policy.disable_training()
 
    def train(self, rounds: int) -> None:
        policy_losses, qf_losses = [], []
        for n in (pbar := tqdm(range(rounds))):
            self.fill_buffer(self.episodes_per_update)
            policy_loss, qf_loss = self.update(self.batches_per_update)
            policy_losses.extend(policy_loss)
            qf_losses.extend(qf_loss)
            self.exploration_noise.update(n)
            pbar.set_description(f"P: {sum(policy_loss) / len(policy_loss):+4.4f} C: {sum(qf_loss) / len(qf_loss):+4.4f}", refresh=False)
 
    def _update_target_quantile_function(self):
        for target_p, new_p in zip(self._target_quantile_function.parameters(), self.quantile_function.parameters()):
            target_p.data.copy_(target_p.data * self.rho_critic + new_p.data * (1 - self.rho_critic))
           
    def _update_target_policy(self):
        for target_p, new_p in zip(self._target_policy.parameters(), self.policy.parameters()):
            target_p.data.copy_(target_p.data * self.rho_policy + new_p.data * (1 - self.rho_policy))
 
    def fill_buffer(self, n_episodes: int) -> None:
        with torch.no_grad():
            for _ in range(n_episodes):
                state = self.mdp.reset()
                for t in range(self.mdp.nr_steps):
                    self._state_buffer[:, t, self._episode_ptr, :] = state
                    exploration_noise = (
                        torch.randn((self.mdp.nr_paths, self.policy.action_dim))
                        * self.exploration_noise.get_noise()
                    )
                    action = (self.policy.act(state) + exploration_noise).clamp(self.mdp.action_lb, self.mdp.action_ub)
                    self._action_buffer[:, t, self._episode_ptr, :] = action
                    state, reward = self.mdp.step(action)
                    self._reward_buffer[:, t, self._episode_ptr] = reward

                if self.normalise_rewards:
                    batch = self._reward_buffer[:, :, self._episode_ptr]
                    batch_mean = batch.mean()
                    batch_mean_2 = batch.square().mean()
                    batch_count = self.mdp.nr_paths * self.mdp.nr_steps
                    tot = self._reward_count + batch_count

                    new_mean = (
                        self._reward_count * self._reward_mean 
                        + batch_count * batch_mean 
                    ) / tot

                    new_mean_2 = (
                        self._reward_count * self._reward_2_mean
                        + batch_count * batch_mean_2
                    ) / tot
                    self._reward_mean = new_mean
                    self._reward_2_mean = new_mean_2
                    self._reward_count = tot
                self._nr_filled = min(self._nr_filled + 1, self.episodes_per_buffer)
                self._episode_ptr = (self._episode_ptr + 1) % self.episodes_per_buffer
 
    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_idx = torch.randint(0, self.mdp.nr_paths, (self.batch_size,))
        time_idx = torch.randint(0, self.mdp.nr_steps, (self.batch_size,))
        episode_idx = torch.randint(0, self._nr_filled, (self.batch_size,))
 
        states = self._state_buffer[path_idx, time_idx, episode_idx]
        states_next = self._state_buffer[path_idx, (time_idx + 1) % self.mdp.nr_steps, episode_idx]
        actions = self._action_buffer[path_idx, time_idx, episode_idx]
        rewards = self._reward_buffer[path_idx, time_idx, episode_idx]
        if self.normalise_rewards:
            std = (self._reward_2_mean - self._reward_mean ** 2).sqrt().clamp_min(1e-6)
            rewards = (rewards - self._reward_mean) / std
        terminal = self._is_terminal[path_idx, time_idx, episode_idx].float()
 
        with torch.no_grad():
            actions_next = self._target_policy.act(states_next)
    
            targets = (
                rewards[:, torch.newaxis]
                + self.gamma * (1.0 - terminal[:, torch.newaxis]) * self._target_quantile_function.compute_quantiles(
                    states_next,
                    actions_next
                )
            )
 
        return states, actions, targets
 
    @abstractmethod
    def update(self, nr_batches: int) -> list[float]:
        raise NotImplementedError
 
@dataclass
class D4PG_QR(D4PG_Base):
    mse_bound: float | None = None
 
    def update(self, nr_batches: int) -> tuple[list[float], list[float]]:
        policy_losses, qf_losses = [], []
        for batch_nr in range(1, nr_batches + 1):
            states, actions, targets = self._get_batch()
 
            qf_loss = quantile_huber_loss(
                self.quantile_function,
                states,
                actions,
                targets,
                self.mse_bound
            )
            self.quantile_function_optimiser.zero_grad()
            qf_loss.backward()
            if self.critic_max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.quantile_function.parameters(),
                    self.critic_max_gradient_norm
                )
            self.quantile_function_optimiser.step()
            self._update_target_quantile_function()
            qf_losses.append(qf_loss.item())

            if batch_nr % self.policy_update_delay == 0:
                self.quantile_function.disable_training()
                policy_actions = self.policy.act(states)
                policy_loss = -torch.mean(
                    self.objective_function(
                        self.quantile_function.compute_quantiles(states, policy_actions),
                        self.quantile_function
                    )
                )
                self.policy_optimiser.zero_grad()
                policy_loss.backward()
                if self.policy_max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.policy_max_gradient_norm
                    )
                self.policy_optimiser.step()
                self.quantile_function.enable_training()
                self._update_target_policy()
                policy_losses.append(policy_loss.item())

        return policy_losses, qf_losses
       
@dataclass
class D4PG_GQR(D4PG_Base):
    mse_bound_decay: float | None = None
    loss_type: Literal["GQR", "AGQR"] = "GQR"
 
    def __post_init__(self):
        super().__post_init__()
        assert self.loss_type in ["GQR", "AGQR"]
 
    def update(self, nr_batches: int) -> tuple[list[float], list[float]]:
        policy_losses, qf_losses = [], []
        for batch_nr in range(1, nr_batches + 1):
            states, actions, targets = self._get_batch()
            qf_loss = generalised_quantile_huber_loss(
                self.quantile_function,
                states,
                actions,
                targets,
                self.mse_bound_decay,
                approximate=self.loss_type == "AGQR"
            )
            self.quantile_function_optimiser.zero_grad()
            qf_loss.backward()
            if self.critic_max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.quantile_function.parameters(),
                    self.critic_max_gradient_norm
                )
            self.quantile_function_optimiser.step()
            self._update_target_quantile_function()
            qf_losses.append(qf_loss.item())

            if batch_nr % self.policy_update_delay == 0:
                self.quantile_function.disable_training()
                policy_actions = self.policy.act(states)
                policy_loss = -torch.mean(
                    self.objective_function(
                        self.quantile_function.compute_quantiles(states, policy_actions),
                        self.quantile_function
                    )
                )
                self.policy_optimiser.zero_grad()
                policy_loss.backward()
                if self.policy_max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.policy_max_gradient_norm
                    )
                self.policy_optimiser.step()
                self.quantile_function.enable_training()
                self._update_target_policy()
                policy_losses.append(policy_loss.item())
            
        return policy_losses, qf_losses
    
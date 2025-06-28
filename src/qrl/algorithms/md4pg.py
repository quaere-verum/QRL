import torch
from dataclasses import dataclass
import copy as cp
from ..mdp import MDP
from ..functions import DeterministicPolicy, ValueFunction
from typing import Callable
from tqdm import tqdm
import math
SQRT_TWO = math.sqrt(2)
SQRT_TWO_PI = math.sqrt(2 / math.pi)
SQRT_TWOPI = math.sqrt(2 * math.pi)

@dataclass
class ExplorationNoise:
    initial_value: torch.Tensor
    update_frequency: int
    decay_factor: float
    minimum_value: torch.Tensor
 
    def __post_init__(self):
        self._value = self.initial_value
 
    def update(self, n: int) -> None:
        if n % self.update_frequency == 0:
            self._value = torch.maximum(
                self._value * self.decay_factor,
                self.minimum_value
            )
 
    def get_noise(self) -> torch.Tensor:
        return self._value

@dataclass
class MD4PG:
    mdp: MDP
    policy: DeterministicPolicy
    objective_function: Callable[[torch.Tensor], torch.Tensor]
    first_moment_function: ValueFunction
    second_moment_function: ValueFunction
    policy_optimiser: torch.optim.Optimizer
    critic_optimiser: torch.optim.Optimizer
    exploration_noise: ExplorationNoise
    episodes_per_buffer: int
    episodes_per_update: int
    batches_per_update: int
    policy_update_delay: int
    gamma: float
    rho_policy: float
    rho_critic: float
    batch_size: int
    normalise_rewards: bool
    policy_max_gradient_norm: float | None
    critic_max_gradient_norm: float | None
 
    def __post_init__(self):
        assert 0 < self.rho_policy < 1 and 0 < self.rho_critic < 1
        assert self.exploration_noise.initial_value.ndim == self.exploration_noise.minimum_value.ndim == 1
        assert self.exploration_noise.initial_value.shape[0] == self.mdp.action_dim
        assert self.exploration_noise.initial_value.shape == self.exploration_noise.minimum_value.shape
        assert self.policy_update_delay > 0 and self.batches_per_update % self.policy_update_delay == 0
 
        self._episode_ptr = 0
        self._nr_filled = 0
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
 
        self._target_first_moment_function = cp.deepcopy(self.first_moment_function)
        self._target_second_moment_function = cp.deepcopy(self.second_moment_function)
        self._target_policy = cp.deepcopy(self.policy)
        self._target_first_moment_function.disable_training()
        self._target_second_moment_function.disable_training()
        self._target_policy.disable_training()
 
    def train(self, rounds: int) -> None:
        policy_losses, critic_losses = [], []
        for n in (pbar := tqdm(range(rounds))):
            self.fill_buffer(self.episodes_per_update)
            policy_loss, critic_loss = self.update(self.batches_per_update)
            policy_losses.extend(policy_loss)
            critic_losses.extend(critic_loss)
            self.exploration_noise.update(n)
            pbar.set_description(f"P: {sum(policy_loss) / len(policy_loss):.4f} C: {sum(critic_loss) / len(critic_loss):.4f}")
 
    def _update_target_critic(self):
        target_params = list(self._target_first_moment_function.parameters())
        new_params = list(self.first_moment_function.parameters())
        for target_p, new_p in zip(target_params, new_params):
            target_p.data.copy_(target_p.data * self.rho_critic + new_p.data * (1 - self.rho_critic))

        target_params = list(self._target_second_moment_function.parameters())
        new_params = list(self.second_moment_function.parameters())
        for target_p, new_p in zip(target_params, new_params):
            target_p.data.copy_(target_p.data * self.rho_critic + new_p.data * (1 - self.rho_critic))
           
    def _update_target_policy(self):
        target_params = list(self._target_policy.parameters())
        new_params = list(self.policy.parameters())
        for target_p, new_p in zip(target_params, new_params):
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
                self._nr_filled = min(self._nr_filled + 1, self.episodes_per_buffer)
                self._episode_ptr = (self._episode_ptr + 1) % self.episodes_per_buffer
 
    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path_idx = torch.randint(0, self.mdp.nr_paths, (self.batch_size,))
        time_idx = torch.randint(0, self.mdp.nr_steps, (self.batch_size,))
        episode_idx = torch.randint(0, self._nr_filled, (self.batch_size,))
 
        states = self._state_buffer[path_idx, time_idx, episode_idx]
        states_next = self._state_buffer[path_idx, (time_idx + 1) % self.mdp.nr_steps, episode_idx]
        actions = self._action_buffer[path_idx, time_idx, episode_idx]
        rewards = self._reward_buffer[path_idx, time_idx, episode_idx]
        if self.normalise_rewards:
            rewards = (rewards - self._reward_buffer[:, :, :self._nr_filled].mean()) / self._reward_buffer[:, :, :self._nr_filled].std().clamp(min=1e-6, max=None)
        terminal = self._is_terminal[path_idx, time_idx, episode_idx].float()
 
        actions_next = self._target_policy.act(states_next)
 
        first_moment_targets = (
            rewards[:, torch.newaxis]
            + self.gamma * (1.0 - terminal[:, torch.newaxis]) * self._target_first_moment_function.value(
                states_next,
                actions_next
            )
        )
        second_moment_targets = (
            rewards[:, torch.newaxis] ** 2
            + self.gamma ** 2 * (1.0 - terminal[:, torch.newaxis]) * (
                self._target_second_moment_function.value(
                    states_next,
                    actions_next
                )
                + 2 * self.gamma * rewards[:, torch.newaxis] * self._target_first_moment_function.value(
                    states_next,
                    actions_next
                )
            )
        )
 
        return states, actions, first_moment_targets, second_moment_targets
 
    def update(self, nr_batches: int) -> list[float]:
        policy_losses, critic_losses = [], []
        for batch_nr in range(1, nr_batches + 1):
            states, actions, first_moment_targets, second_moment_targets = self._get_batch()
 
            first_moment_estimates = self.first_moment_function.value(states, actions)
            first_moment_function_loss = torch.mean(torch.pow(first_moment_estimates - first_moment_targets, 2))
            second_moment_estimates = self.second_moment_function.value(states, actions)
            second_moment_function_loss = torch.mean(torch.pow(second_moment_estimates - second_moment_targets, 2))
            critic_loss = first_moment_function_loss + second_moment_function_loss
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            if self.critic_max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.first_moment_function.parameters(),
                    self.critic_max_gradient_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.second_moment_function.parameters(),
                    self.critic_max_gradient_norm
                )
            self.critic_optimiser.step()
            self._update_target_critic()
            critic_losses.append(critic_loss.item())

            if batch_nr % self.policy_update_delay == 0:
                self.first_moment_function.disable_training()
                self.second_moment_function.disable_training()
                policy_actions = self.policy.act(states)
                policy_loss = -torch.mean(
                    self.objective_function(
                        self.first_moment_function.value(states, policy_actions),
                        self.second_moment_function.value(states, policy_actions)
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
                self.first_moment_function.enable_training()
                self.second_moment_function.enable_training()
                self._update_target_policy()
                policy_losses.append(policy_loss.item())

        return policy_losses, critic_losses
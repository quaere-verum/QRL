import torch
import torch.nn.functional as F
from dataclasses import dataclass
import copy as cp
from ..mdp import MDP
from ..functions import DeterministicPolicy, ValueFunction, GaussianNoiseScale
from tqdm import tqdm
    
@dataclass 
class TD3:
    mdp: MDP
    policy: DeterministicPolicy
    value_function_1: ValueFunction
    value_function_2: ValueFunction
    policy_optimiser: torch.optim.Optimizer
    value_functions_optimiser: torch.optim.Optimizer
    exploration_noise: GaussianNoiseScale
    policy_smoothing_std: float
    policy_smoothing_clip_range: float
    policy_update_delay: int
    episodes_per_buffer: int
    episodes_per_update: int
    batches_per_update: int
    gamma: float
    rho_policy: float
    rho_critic: float
    batch_size: int
    normalise_rewards: bool = False
    policy_max_gradient_norm: float | None = None
    critic_max_gradient_norm: float | None = None

    def __post_init__(self):
        assert self.batches_per_update % self.policy_update_delay == 0
        assert 0 <= self.rho_policy < 1 and 0 <= self.rho_critic < 1
        assert self.exploration_noise.initial_value.ndim == self.exploration_noise.minimum_value.ndim == 1
        assert self.exploration_noise.initial_value.shape[0] == self.mdp.action_dim
        assert self.exploration_noise.initial_value.shape == self.exploration_noise.minimum_value.shape
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

        self._target_value_function_1 = cp.deepcopy(self.value_function_1)
        self._target_value_function_2 = cp.deepcopy(self.value_function_2)
        self._target_policy = cp.deepcopy(self.policy)
        self._target_value_function_1.disable_training()
        self._target_value_function_2.disable_training()
        self._target_policy.disable_training()

    def train(self, rounds: int) -> None:
        policy_losses, critic_losses = [], []
        for n in (pbar := tqdm(range(rounds))):
            self.fill_buffer(self.episodes_per_update)
            policy_loss, critic_loss = self.update(self.batches_per_update)
            policy_losses.extend(policy_loss)
            critic_losses.extend(critic_loss)
            self.exploration_noise.update(n)
            pbar.set_description(f"P: {sum(policy_loss) / len(policy_loss):+4.4f} C: {sum(critic_loss) / len(critic_loss):+4.4f}", refresh=False)
        return

    def _update_target_value_functions(self):
        for target_p, new_p in zip(self._target_value_function_1.parameters(), self.value_function_1.parameters()):
            target_p.data.copy_(target_p.data * self.rho_critic + new_p.data * (1 - self.rho_critic))
        
        for target_p, new_p in zip(self._target_value_function_2.parameters(), self.value_function_2.parameters()):
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
        time_idx = torch.randint(0, self.mdp.nr_steps - 1, (self.batch_size,)) # Exclude terminal state because it doesn't need to be bootstrapped
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
            smoothing_noise = (
                (torch.randn_like(actions_next) * self.policy_smoothing_std)
                .clamp(-self.policy_smoothing_clip_range, self.policy_smoothing_clip_range)
            )
            actions_next = (actions_next + smoothing_noise).clip(self.mdp.action_lb, self.mdp.action_ub)

            value_next_1 = self._target_value_function_1.value(
                states_next, actions_next
            ).flatten()
            value_next_2 = self._target_value_function_2.value(
                states_next, actions_next
            ).flatten()

            targets = (
                rewards
                + self.gamma * (1.0 - terminal) * torch.min(value_next_1, value_next_2)
            ) 

        return states, actions, targets

    def update(self, nr_batches: int) -> tuple[list[float], list[float]]:
        policy_losses, critic_losses = [], []
        for batch_nr in range(1, nr_batches + 1):
            states, actions, targets = self._get_batch()
            
            value_estimates_1 = self.value_function_1.value(states, actions).flatten()
            value_function_loss_1 = F.smooth_l1_loss(value_estimates_1, targets)
            value_estimates_2 = self.value_function_2.value(states, actions).flatten()
            value_function_loss_2 = F.smooth_l1_loss(value_estimates_2, targets)
            critic_loss = value_function_loss_1 + value_function_loss_2
            self.value_functions_optimiser.zero_grad()
            critic_loss.backward()
            if self.critic_max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.value_function_1.parameters(),
                    self.critic_max_gradient_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_function_2.parameters(),
                    self.critic_max_gradient_norm
                )
            self.value_functions_optimiser.step()
            self._update_target_value_functions()
            critic_losses.append(critic_loss.item())

            if batch_nr % self.policy_update_delay == 0:
                self.value_function_1.disable_training()
                policy_actions = self.policy.act(states)
                policy_loss = -torch.mean(self.value_function_1.value(states, policy_actions))
                self.policy_optimiser.zero_grad()
                policy_loss.backward()
                if self.policy_max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.policy_max_gradient_norm
                    )
                self.policy_optimiser.step()
                self.value_function_1.enable_training()
                self._update_target_policy()
                policy_losses.append(policy_loss.item())
        return policy_losses, critic_losses

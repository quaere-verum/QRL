import torch
import torch.nn.functional as F
from dataclasses import dataclass
import copy as cp
from ..mdp import MDP
from ..functions import DiscreteValueFunction
from tqdm import tqdm


@dataclass
class DDQN:
    mdp: MDP
    value_function: DiscreteValueFunction
    value_function_optimiser: torch.optim.Optimizer
    episodes_per_buffer: int
    episodes_per_update: int
    batches_per_update: int
    gamma: float
    rho_critic: float
    batch_size: int
    epsilon: float = 1.0
    epsilon_decay_factor: float = 0.99
    minimum_epsilon: float = 0.05
    normalise_rewards: bool = False
    critic_max_gradient_norm: float | None = None

    def __post_init__(self):
        assert 0 <= self.rho_critic < 1
        assert 0 <= self.epsilon <= 1 and 0 <= self.minimum_epsilon <= 1
        assert 0 <= self.epsilon_decay_factor <= 1
        self._episode_ptr = 0
        self._nr_filled = 0
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_2_mean = 0.0
        self._state_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps + 1, self.episodes_per_buffer, self.mdp.state_dim),
            dtype=torch.float32,
        )
        self._action_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer, 1),
            dtype=torch.int64,
        )
        self._reward_buffer = torch.zeros(
            (self.mdp.nr_paths, self.mdp.nr_steps, self.episodes_per_buffer),
            dtype=torch.float32,
        )
        self._is_terminal = torch.zeros_like(self._reward_buffer, dtype=torch.bool)
        self._is_terminal[:, -1, :] = True
        self._target_value_function = cp.deepcopy(self.value_function)
        self._target_value_function.disable_training()

    @property
    def policy(self) -> DiscreteValueFunction:
        return self.value_function

    def train(self, rounds: int) -> None:
        for _ in (pbar := tqdm(range(rounds))):
            self.fill_buffer(self.episodes_per_update)
            critic_losses = self.update(self.batches_per_update)
            self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay_factor)
            pbar.set_description(f"C: {sum(critic_losses) / len(critic_losses):+4.4f}", refresh=False)

    def _update_target_value_function(self):
        for target_p, new_p in zip(self._target_value_function.parameters(), self.value_function.parameters()):
            target_p.data.copy_(target_p.data * self.rho_critic + new_p.data * (1 - self.rho_critic))

    def fill_buffer(self, n_episodes: int) -> None:
        with torch.no_grad():
            for _ in range(n_episodes):
                state = self.mdp.reset()
                self._state_buffer[:, 0, self._episode_ptr, :] = state
                for t in range(self.mdp.nr_steps):
                    action = self.policy.act(state, epsilon=self.epsilon)
                    self._action_buffer[:, t, self._episode_ptr, :] = action
                    state, reward = self.mdp.step(action)
                    self._state_buffer[:, t + 1, self._episode_ptr, :] = state
                    self._reward_buffer[:, t, self._episode_ptr] = reward

                if self.normalise_rewards:
                    batch = self._reward_buffer[:, :, self._episode_ptr]
                    batch_count = self.mdp.nr_paths * self.mdp.nr_steps
                    tot = self._reward_count + batch_count
                    self._reward_mean = (
                        self._reward_count * self._reward_mean + batch_count * batch.mean()
                    ) / tot
                    self._reward_2_mean = (
                        self._reward_count * self._reward_2_mean + batch_count * batch.square().mean()
                    ) / tot
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
            std = (self._reward_2_mean - self._reward_mean ** 2).clamp_min(0).sqrt().clamp_min(1e-6)
            rewards = (rewards - self._reward_mean) / std
        terminal = self._is_terminal[path_idx, time_idx, episode_idx].float()

        with torch.no_grad():
            actions_next = self.value_function.act(states_next)

            targets = (
                rewards
                + self.gamma * (1.0 - terminal) * self._target_value_function.value(
                    states_next,
                    actions_next
                ).flatten()
            )

        return states, actions, targets

    def update(self, nr_batches: int) -> list[float]:
        critic_losses = []
        for _ in range(nr_batches):
            states, actions, targets = self._get_batch()
            value_estimates = self.value_function.value(states, actions).flatten()
            value_function_loss = F.smooth_l1_loss(value_estimates, targets)
            self.value_function_optimiser.zero_grad()
            value_function_loss.backward()
            if self.critic_max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.value_function.parameters(), self.critic_max_gradient_norm
                )
            self.value_function_optimiser.step()
            self._update_target_value_function()
            critic_losses.append(value_function_loss.item())
        return critic_losses

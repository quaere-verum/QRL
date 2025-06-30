import torch
from dataclasses import dataclass
from ..mdp import MDP
from ..functions import GaussianPolicy, StateValueFunction
from tqdm import tqdm
import copy as cp

@dataclass 
class VMPO:
    mdp: MDP
    policy: GaussianPolicy
    value_function: StateValueFunction
    policy_optim: torch.optim.Optimizer
    value_function_optim: torch.optim.Optimizer
    batch_size: int
    update_per_rollout: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float | None = 1.0

    def __post_init__(self):
        self._eps_alpha = 0.1
        self._eps_eta = 0.01
        self._eta = torch.nn.Parameter(torch.tensor([0.05]), requires_grad=True)
        self._alpha = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self._multiplier_optimiser = torch.optim.Adam(
            [{"params": [self._eta, self._alpha]}], lr=0.01
        )
        self._target_policy = cp.deepcopy(self.policy)
        self._target_policy.disable_training()

    def _update_target_policy(self):
        target_params = list(self._target_policy.parameters())
        new_params = list(self.policy.parameters())
        for target_p, new_p in zip(target_params, new_params):
            target_p.data.copy_(new_p.data)

    def compute_gae(self, rewards: torch.Tensor, state_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)    
        gae = 0
        for t in reversed(range(self.mdp.nr_steps)):
            if t == self.mdp.nr_steps - 1:
                next_value = 0
                next_done = True
            else:
                next_value = state_values[:, t + 1]
                next_done = False
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - next_done) - state_values[:, t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[:, t] = gae
            
        scores = advantages + state_values
        return advantages, scores

    def collect_rollout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        states = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps, self.mdp.state_dim), dtype=torch.float32)
        state_values = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        actions = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps, self.mdp.action_dim), dtype=torch.float32)

        state = self.mdp.reset()
        with torch.no_grad():
            for t in range(self.mdp.nr_steps):
                state_values[:, t] = self.value_function.value(state).flatten()
                action = self._target_policy.act(state)
                actions[:, t, :] = action
                states[:, t, :] = state
                state, reward = self.mdp.step(action)
                rewards[:, t] = reward
        rewards = (rewards - rewards.mean()) / rewards.std().clamp(1e-6, None)
        advantages, scores = self.compute_gae(
            rewards,
            state_values
        )

        return states, actions, advantages, scores
    

    def train(self, n: int) -> None:
        for round in (pbar := tqdm(range(n))):
            states, actions, advantages, scores = self.collect_rollout()
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6, max=None)
            policy_losses, value_losses, entropy_losses = [], [], []
            for update_nr in range(self.update_per_rollout):
                path_idx = torch.randint(0, self.mdp.nr_paths, (self.batch_size,))
                time_idx = torch.randint(0, self.mdp.nr_steps, (self.batch_size,))

                states_batch = states[path_idx, time_idx]
                actions_batch = actions[path_idx, time_idx]
                advantages_batch = advantages[path_idx, time_idx]
                scores_batch = scores[path_idx, time_idx]

                logprobs_new = self.policy.get_logprobs(states_batch, actions_batch)

                kl_divergence = torch.distributions.kl_divergence(
                    self._target_policy.get_dist(states_batch),
                    self.policy.get_dist(states_batch)
                )
                entropy_loss = (
                    torch.clamp(self._alpha, 0.0) * (self._eps_alpha - kl_divergence.detach())
                    + torch.clamp(self._alpha.detach(), 0.0) * kl_divergence
                ).mean()

                top_indices = torch.sort(advantages_batch, descending=True).indices[:len(advantages_batch) // 2]
                top_advantages = advantages_batch[top_indices]

                weights = ((top_advantages - top_advantages.max()) / self._eta.detach()).exp()
                weights = weights / weights.sum()

                policy_loss = -(weights * logprobs_new[top_indices]).sum()

                temperature_loss = (
                    self._eta * self._eps_eta
                    + self._eta * ((advantages_batch - advantages_batch.max()) / self._eta).exp().mean().log()
                )

                values = self.value_function.value(states_batch).flatten()
                value_loss = torch.nn.functional.mse_loss(scores_batch, values)

                self.policy_optim.zero_grad()
                self.value_function_optim.zero_grad()
                self._multiplier_optimiser.zero_grad()
                
                policy_loss.backward()
                value_loss.backward()
                entropy_loss.backward()
                temperature_loss.backward()
                
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
                
                self.policy_optim.step()
                self.value_function_optim.step()
                self._multiplier_optimiser.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

            self._update_target_policy()
            pbar.set_description(
                f"P: {sum(policy_losses) / len(policy_losses):.4f} C: {sum(value_losses) / len(value_losses):.4f} E: {sum(entropy_losses) / len(entropy_losses):.4f}"
            )
        return
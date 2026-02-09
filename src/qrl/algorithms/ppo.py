import torch
from dataclasses import dataclass
from ..mdp import MDP
from ..functions import GaussianPolicy, StateValueFunction
from tqdm import tqdm

@dataclass 
class PPO:
    mdp: MDP
    policy: GaussianPolicy
    value_function: StateValueFunction
    policy_optim: torch.optim.Optimizer
    value_function_optim: torch.optim.Optimizer
    batch_size: int
    update_per_rollout: int
    gamma: float = 0.99
    ent_coef: float = 1e-2
    gae_lambda: float | None = None
    clip_range: float = 0.2
    max_grad_norm: float | None = 1.0
    
    def compute_advantages_scores(self, rewards: torch.Tensor, state_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gae_lambda is None:
            scores = torch.zeros_like(rewards)
            for t in reversed(range(self.mdp.nr_steps)):
                if t == self.mdp.nr_steps - 1:
                    scores[:, t] = rewards[:, t]
                else:
                    scores[:, t] = rewards[:, t] + self.gamma * scores[:, t + 1]
            advantages = scores - state_values
        else:
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

    def collect_rollout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        rewards = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        states = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps, self.mdp.state_dim), dtype=torch.float32)
        state_values = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        actions = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps, self.mdp.action_dim), dtype=torch.float32)

        state = self.mdp.reset()
        with torch.no_grad():
            for t in range(self.mdp.nr_steps):
                state_values[:, t] = self.value_function.value(state).flatten()
                action, logprob = self.policy.act(state, return_logprobs=True)
                actions[:, t, :] = action
                logprobs[:, t] = logprob.flatten()
                states[:, t, :] = state
                state, reward = self.mdp.step(action)
                rewards[:, t] = reward
        
        advantages, scores = self.compute_advantages_scores(rewards, state_values)

        return states, actions, advantages, scores, logprobs
    

    def train(self, n: int) -> list[float]:
        for round in (pbar := tqdm(range(n))):
            states, actions, advantages, scores, logprobs = self.collect_rollout()
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(min=1e-6, max=None)
            
            policy_losses, value_losses = [], []
            for update_nr in range(self.update_per_rollout):
                path_idx = torch.randint(0, self.mdp.nr_paths, (self.batch_size,))
                time_idx = torch.randint(0, self.mdp.nr_steps, (self.batch_size,))

                states_batch = states[path_idx, time_idx]
                actions_batch = actions[path_idx, time_idx]
                advantages_batch = advantages[path_idx, time_idx]
                scores_batch = scores[path_idx, time_idx]
                logprobs_batch = logprobs[path_idx, time_idx]

                logprobs_new, entropy_new = self.policy.get_logprobs(states_batch, actions_batch, return_entropy=True)
                logprobs_new = logprobs_new.flatten()
                ratio = torch.exp((logprobs_new - logprobs_batch).clamp(-20.0, 20.0))
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() - self.ent_coef * entropy_new.mean()

                values = self.value_function.value(states_batch).flatten()
                value_loss = torch.nn.functional.mse_loss(scores_batch, values)

                self.policy_optim.zero_grad()
                policy_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()
                policy_losses.append(policy_loss.item())

                self.value_function_optim.zero_grad()
                value_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
                self.value_function_optim.step()
                value_losses.append(value_loss.item())
            pbar.set_description(f"P: {sum(policy_losses) / len(policy_losses):+4.4f} C: {sum(value_losses) / len(value_losses):+4.4f}", refresh=False)
            
        return
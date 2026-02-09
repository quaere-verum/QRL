import torch
from dataclasses import dataclass
from ..mdp import MDP
from ..functions import GaussianPolicy
from tqdm import tqdm

@dataclass
class REINFORCE:
    mdp: MDP
    policy: GaussianPolicy
    optimiser: torch.optim.Optimizer
    gamma: float
    score_threshold: float | None = None
    max_grad_norm: float | None = 1.0


    def compute_scores(self, rewards: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros_like(rewards)
        for t in reversed(range(self.mdp.nr_steps)):
            if t == self.mdp.nr_steps - 1:
                scores[:, t] = rewards[:, t]
            else:
                scores[:, t] = rewards[:, t] + self.gamma * scores[:, t + 1]
        return scores

    def train(self, n: int):
        for round in (pbar := tqdm(range(n))):
            logprobs, scores = self.collect_rollout()
            if self.score_threshold is not None:
                if scores[:, 0].mean().item() >= self.score_threshold:
                    break
            weighted_logprobs = logprobs * scores
            loss = -torch.mean(weighted_logprobs)
            self.optimiser.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimiser.step()
            pbar.set_description(f"P: {loss.item():+4.4f}", refresh=False)


    def collect_rollout(self) -> tuple[torch.Tensor, torch.Tensor]:
        logprobs = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)
        rewards = torch.zeros((self.mdp.nr_paths, self.mdp.nr_steps), dtype=torch.float32)

        state = self.mdp.reset()
        for t in range(self.mdp.nr_steps):
            action, logprob = self.policy.act(state, return_logprobs=True)
            logprobs[:, t] = logprob.flatten()
            state, reward = self.mdp.step(action)
            rewards[:, t] = reward
        scores = self.compute_scores(rewards)
        logprobs = logprobs
        return logprobs, scores

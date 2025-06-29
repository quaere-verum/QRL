from __future__ import annotations
import torch
from dataclasses import dataclass
from .functions import Policy
from abc import abstractmethod
from scipy.stats import norm

@dataclass
class MDP:
    action_lb: torch.Tensor
    action_ub: torch.Tensor

    def __post_init__(self):
        assert self.action_lb.ndim == self.action_ub.ndim == 1
        assert self.action_lb.shape[0] == self.action_ub.shape[0] == self.action_dim
        assert torch.all(self.action_lb < self.action_ub)
    
    @property
    def state_dim(self) -> int:
        raise NotImplementedError
    
    @property
    def action_dim(self) -> int:
        raise NotImplementedError
    
    @property
    def nr_paths(self) -> int:
        raise NotImplementedError
    
    @property
    def nr_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, policy: Policy) -> torch.Tensor:
        rewards = []
        actions = []
        state = self.reset()
        with torch.no_grad():
            for t in range(self.nr_steps):
                action = policy.act(state)
                state, reward = self.step(action)
                actions.append(action)
                rewards.append(reward)
        return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)

@dataclass
class LQGMDP(MDP):
    noise_paths: torch.Tensor
    state_transform: dict[int, torch.Tensor]
    control_transform: dict[int, torch.Tensor]
    state_penalty: dict[int, torch.Tensor]
    final_state_penalty: torch.Tensor
    control_penalty: dict[int, torch.Tensor]
    initial_state: torch.Tensor

    def __post_init__(self):
        super().__post_init__()
        for obj in [self.state_transform, self.control_transform, self.state_penalty, self.control_penalty]:
            assert isinstance(obj, dict)
            assert len(obj) == self.nr_steps
            for key in obj.keys(): 
                assert isinstance(key, int)
        for obj in [self.state_transform, self.control_transform, self.state_penalty, self.control_penalty]:
            for time in obj:
                assert obj[time].shape == obj[0].shape
        
        assert self.state_transform[0].ndim == self.control_transform[0].ndim == 2
        assert self.state_transform[0].shape[0] == self.state_transform[0].shape[1]
        assert self.control_transform[0].shape[0] == self.state_transform[0].shape[0]
        assert self.state_transform[0].shape == self.state_penalty[0].shape == self.final_state_penalty.shape
        assert self.control_penalty[0].shape[0] == self.control_penalty[0].shape[1] == self.control_transform[0].shape[1]
        assert self.initial_state.ndim == 1 and self.initial_state.shape[0] == self.state_transform[0].shape[0]

        self._state = torch.zeros((self.nr_paths, self.initial_state.shape[0]), dtype=torch.float32)        

    @property
    def nr_paths(self) -> int:
        return self.noise_paths.shape[0]
    
    @property
    def nr_steps(self) -> int:
        return self.noise_paths.shape[1]

    @property
    def state_dim(self) -> int:
        return self.state_transform[0].shape[0] + 1
    
    @property
    def action_dim(self) -> int:
        return self.control_transform[0].shape[1]
    
    def _get_obs(self) -> torch.Tensor:
        time_remaining = torch.full((self.nr_paths, 1), self._t / self.nr_steps, dtype=torch.float32)
        return torch.concat((self._state, time_remaining), dim=1)
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action.clamp_(self.action_lb, self.action_ub)
        if self._t >= self.nr_steps:
            raise ValueError
        
        self._t += 1
        
        cost = (
            torch.einsum(
                "bi, ij, bj -> b", self._state, self.state_penalty[self._t - 1], self._state
            ) + 
            torch.einsum(
                "bi, ij, bj -> b", action, self.control_penalty[self._t - 1], action
            )
        )

        self._state = (
            torch.matmul(self._state, self.state_transform[self._t - 1].T) +
            torch.matmul(action, self.control_transform[self._t - 1].T) +
            self.noise_paths[:, self._t - 1, torch.newaxis]
        )
        if self._t == self.nr_steps:
            cost += torch.einsum(
                "bi, ij, bj -> b", self._state, self.final_state_penalty, self._state
            )

        return self._get_obs(), -cost

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._state = torch.zeros((self.nr_paths, self.initial_state.shape[0]), dtype=torch.float32)
        self._state += self.initial_state
        return self._get_obs()
    
    def solve_riccati_equation(self) -> dict[int, torch.Tensor]:
        solutions = [self.final_state_penalty.clone()]
        for t in reversed(range(self.nr_steps)):
            solution = (
                self.state_transform[t].T @ (
                    solutions[-1] 
                    - solutions[-1] @ self.control_transform[t] @ torch.linalg.inv(
                        self.control_transform[t].T @ solutions[-1] @ self.control_transform[t]
                        + self.control_penalty[t]
                    ) @ self.control_transform[t].T @ solutions[-1]
                ) @ self.state_transform[t] + self.state_penalty[t]
            )
            solutions.append(solution)
        solutions = dict(zip(range(len(solutions)), solutions[::-1]))
        return solutions
    
    def solve(self) -> tuple[torch.Tensor, torch.Tensor]:
        actions, rewards = [], []
        riccati_solution = self.solve_riccati_equation()
        state = self.reset()[:, :-1]
        for t in range(self.nr_steps):
            feedback_gain_matrix = torch.linalg.inv(
                self.control_transform[t].T @ riccati_solution[t + 1] @ self.control_transform[t]
                + self.control_penalty[t]
            ) @ self.control_transform[t].T @ riccati_solution[t + 1] @ self.state_transform[t]
            action = -torch.matmul(state, feedback_gain_matrix.T)
            state, reward = self.step(action)
            state = state[:, :-1]
            actions.append(action)
            rewards.append(reward)
        return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)

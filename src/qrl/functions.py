from __future__ import annotations
import torch
from abc import abstractmethod
from dataclasses import dataclass

@dataclass
class GaussianNoiseScale:
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

class Policy(torch.nn.Module):
    __slots__ = [
        "state_dim",
        "action_dim",
        "hidden_size",
        "action_lb",
        "action_ub",
    ]
    @abstractmethod
    def act(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianPolicy(torch.nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_size: int,
        action_lb: torch.Tensor,
        action_ub: torch.Tensor,
        min_std: float,
        max_std: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim)
        )
        self.std = torch.nn.Parameter(torch.full((action_dim,), max_std), requires_grad=True)
        self._deterministic_mode = False
        self.min_std = min_std
        self.max_std = max_std
 
    def deterministic(self, deterministic: bool) -> None:
        self._deterministic_mode = deterministic
 
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.clamp(self.model.forward(x), self.action_lb, self.action_ub)
        std = self.std.clamp(self.min_std, self.max_std)
        return mean, std.broadcast_to(mean.shape)
    
    def get_logprobs(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        logprobs = dist.log_prob(actions)
        return torch.sum(logprobs, dim=-1)
    
    def get_dist(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist
 
    def act(
        self,
        state: torch.Tensor,
        *,
        return_logprobs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        if self._deterministic_mode:
            actions = mean.clamp(self.action_lb, self.action_ub)
        else:
            actions = dist.sample().clamp(self.action_lb, self.action_ub)
        if return_logprobs:
            logprobs = torch.sum(dist.log_prob(actions), dim=-1)
            return actions, logprobs
        else:
            return actions
        
    def disable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def enable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)


class StateValueFunction(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
 
    def disable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
           
    def enable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
   
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class DeterministicPolicy(torch.nn.Module):
    def __init__(
        self, 
        action_dim: int, 
        state_dim: int, 
        hidden_size: int, 
        action_lb: torch.Tensor, 
        action_ub: torch.Tensor
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim)
        )

    def disable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    def enable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        actions = self.forward(state).clamp(self.action_lb, self.action_ub)
        return actions

class ValueFunction(torch.nn.Module):
    def __init__(self, action_dim: int, state_dim: int, hidden_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim + self.action_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def disable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
            
    def enable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
    
    def value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.concat((state, action), dim=1)
        return self.forward(x)
    
class QuantileFunction(torch.nn.Module):
    def __init__(
        self,
        q_start: float,
        q_end: float,
        nr_quantiles: int,
        action_dim: int, 
        state_dim: int, 
        hidden_size: int,
    ):
        super().__init__()
        assert 0 <= q_start < q_end <= 1.0, "Quantile locations must be in the range (0, 1)"
        linspace = torch.linspace(q_start, q_end, nr_quantiles + 1)
        self.quantile_locations = (linspace[:-1] + linspace[1:]) / 2.0
        self.probability_mass = linspace.diff()
        self.b = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.nr_quantiles = nr_quantiles
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim + self.action_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, nr_quantiles)
        )

    def disable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
            
    def enable_training(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
    
    def compute_quantiles(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.concat((state, action), dim=1)
        return self.forward(x)
    
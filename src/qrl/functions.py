from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch

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
                self.minimum_value,
            )

    def get_noise(self) -> torch.Tensor:
        return self._value


class Policy(torch.nn.Module, ABC):
    __slots__ = [
        "state_dim",
        "action_dim",
        "hidden_size",
        "action_lb",
        "action_ub",
    ]
    action_dtype = torch.float32

    @abstractmethod
    def act(self, state: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class StochasticPolicy(Policy, ABC):
    def __init__(self):
        super().__init__()
        self._deterministic_mode = False

    @abstractmethod
    def get_dist(self, state: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def deterministic(self, value: bool) -> None:
        self._deterministic_mode = value

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def act(
        self,
        state: torch.Tensor,
        *,
        return_logprobs: bool = False,
    ):
        dist = self.get_dist(state)
        action = dist.mode if self._deterministic_mode else dist.sample()
        action = self._postprocess_action(action)

        if return_logprobs:
            return action, dist.log_prob(action)
        return action

    def get_logprobs(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        return_entropy: bool = False,
    ):
        dist = self.get_dist(state)
        logprobs = dist.log_prob(actions)
        if return_entropy:
            return logprobs, dist.entropy()
        return logprobs

    def disable_training(self) -> None:
        self.requires_grad_(False)

    def enable_training(self) -> None:
        self.requires_grad_(True)


class GaussianPolicy(StochasticPolicy):
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
            torch.nn.Linear(hidden_size, action_dim),
        )
        self.std = torch.nn.Parameter(
            torch.full((action_dim,), max_std), requires_grad=True
        )
        self.min_std = min_std
        self.max_std = max_std

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.model(state).clamp(self.action_lb, self.action_ub)
        std = self.std.clamp(self.min_std, self.max_std)
        return mean, std.expand_as(mean)

    def get_dist(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean, std = self.forward(state)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1
        )

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.clamp(self.action_lb, self.action_ub)


class CategoricalPolicy(StochasticPolicy):
    action_dtype = torch.int64

    def __init__(
        self,
        n_actions: int,
        state_dim: int,
        hidden_size: int,
        *,
        mask_from_state: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = 1  # One integer-valued action per path.
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.action_lb = torch.tensor([0])
        self.action_ub = torch.tensor([n_actions])
        self.mask_from_state = mask_from_state
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions),
        )

    def get_dist(self, state: torch.Tensor) -> torch.distributions.Distribution:
        logits: torch.Tensor = self.model.forward(state)
        if self.mask_from_state:
            mask = state[:, -self.n_actions:] > 0.5
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        return torch.distributions.Independent(
            torch.distributions.Categorical(logits=logits.unsqueeze(1)), 1
        )


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


class DeterministicPolicy(Policy):
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
            torch.nn.Linear(hidden_size, action_dim),
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
            torch.nn.Linear(hidden_size, 1),
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


class DiscreteValueFunction(Policy):
    action_dtype = torch.int64

    def __init__(
        self,
        n_actions: int,
        state_dim: int,
        hidden_size: int,
        *,
        mask_from_state: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = 1
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.action_lb = torch.tensor([0])
        self.action_ub = torch.tensor([n_actions])
        self.mask_from_state = mask_from_state
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_actions),
        )

    def disable_training(self) -> None:
        self.requires_grad_(False)

    def enable_training(self) -> None:
        self.requires_grad_(True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model.forward(state)

    def value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state).gather(1, action)

    def act(self, state: torch.Tensor, *, epsilon: float = 0.0) -> torch.Tensor:
        values = self.forward(state)
        mask = (state[:, -self.n_actions:] > 0.5 if self.mask_from_state
                else torch.ones_like(values, dtype=torch.bool))
        actions = values.masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
        if epsilon > 0:
            random_actions = torch.multinomial(mask.float(), 1)
            explore = torch.rand_like(actions, dtype=torch.float32) < epsilon
            actions = torch.where(explore, random_actions, actions)
        return actions


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

from __future__ import annotations

from importlib.resources import files
from itertools import chain
from math import sqrt
from typing import Protocol

import torch
import yaml
from torch import Tensor
from torch.optim import Adam

from ..functions import (
    CategoricalPolicy,
    DeterministicPolicy,
    GaussianNoiseScale,
    GaussianPolicy,
    Policy,
    QuantileFunction,
    StateValueFunction,
    ValueFunction,
)
from ..mdp import MDP
from .d4pg_qr import D4PG_GQR, D4PG_QR
from .ddpg import DDPG
from .md4pg import MD4PG
from .ppo import PPO
from .reinforce import REINFORCE
from .td3 import TD3
from .vmpo import VMPO

__all__ = [
    "MD4PG", 
    "D4PG_QR", 
    "D4PG_GQR", 
    "DDPG", 
    "PPO", 
    "REINFORCE",
    "TD3", 
    "VMPO", 
    "get_algorithm",
]


class Algorithm(Protocol):
    mdp: MDP
    policy: Policy

    def train(self, rounds: int) -> None: ...


def _stochastic_policy(mdp: MDP, config: dict, policy_type: str, masked: bool):
    common = dict(state_dim=mdp.state_dim, hidden_size=config["policy_hidden_size"])
    if policy_type == "categorical":
        upper = mdp.action_ub.item() if mdp.action_ub.numel() == 1 else None
        if (mdp.action_dim != 1 or mdp.action_lb.numel() != 1
                or mdp.action_lb.item() != 0 or upper is None
                or upper < 1 or upper != int(upper)):
            raise ValueError("Categorical actions require bounds [0], [N] and action_dim=1")
        return CategoricalPolicy(
            **common, n_actions=int(upper), mask_from_state=masked
        )
    if policy_type == "gaussian":
        return GaussianPolicy(
            **common, 
            action_dim=mdp.action_dim,
            action_lb=mdp.action_lb, 
            action_ub=mdp.action_ub,
            min_std=config["policy_min_std"], 
            max_std=config["policy_max_std"],
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


def _exploration_noise(mdp: MDP, config: dict, *, normalise: bool = True):
    size = mdp.action_dim if normalise else 1
    scale = sqrt(mdp.action_dim) if normalise else 1.0
    return GaussianNoiseScale(
        initial_value=torch.full((size,), config["initial_exploration_noise"] / scale),
        minimum_value=torch.full((size,), config["minimum_exploration_noise"] / scale),
        update_frequency=config["noise_update_frequency"],
        decay_factor=config["noise_decay_factor"],
    )


def _quantile_objective(quantiles: Tensor, critic: QuantileFunction) -> Tensor:
    return (quantiles * critic.probability_mass[None, :]).sum(dim=1)


def get_algorithm(
    algorithm_name: str,
    mdp: MDP,
    *,
    policy_type: str = "gaussian",
    mask_from_state: bool = False,
    **overrides,
) -> Algorithm:
    name = algorithm_name.lower()
    if name == "md4pg":
        raise NotImplementedError("MD4PG is not yet available from the factory")
    if name not in {"reinforce", "ppo", "vmpo", "ddpg", "td3", "d4pg_qr", "d4pg_gqr"}:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    if policy_type not in {"gaussian", "categorical"}:
        raise ValueError(f"Unknown policy type: {policy_type}")
    if name not in {"reinforce", "ppo", "vmpo"} and policy_type != "gaussian":
        raise ValueError(f"{name} only supports continuous actions")

    path = files("qrl.config").joinpath(f"{name}.yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    kwargs = {**config["kwargs"], **overrides}

    if name in {"reinforce", "ppo", "vmpo"}:
        policy = _stochastic_policy(mdp, config, policy_type, mask_from_state)
        if name == "reinforce":
            return REINFORCE(
                mdp=mdp, policy=policy,
                optimiser=Adam(policy.parameters(), lr=config["lr"]),
                **kwargs,
            )
        critic = StateValueFunction(mdp.state_dim, config["critic_hidden_size"])
        return (PPO if name == "ppo" else VMPO)(
            mdp=mdp, 
            policy=policy, 
            value_function=critic,
            policy_optim=Adam(policy.parameters(), lr=config["policy_lr"]),
            value_function_optim=Adam(critic.parameters(), lr=config["critic_lr"]),
            **kwargs,
        )

    policy = DeterministicPolicy(
        action_dim=mdp.action_dim,
        state_dim=mdp.state_dim,
        hidden_size=config["policy_hidden_size"],
        action_lb=mdp.action_lb, 
        action_ub=mdp.action_ub,
    )
    common = dict(
        mdp=mdp, policy=policy,
        policy_optimiser=Adam(policy.parameters(), lr=config["policy_lr"]),
        exploration_noise=_exploration_noise(
            mdp, config, normalise=name != "d4pg_gqr"
        ),
        **kwargs,
    )

    if name == "ddpg":
        critic = ValueFunction(mdp.action_dim, mdp.state_dim, config["critic_hidden_size"])
        return DDPG(
            **common, value_function=critic,
            value_function_optimiser=Adam(critic.parameters(), lr=config["critic_lr"]),
        )
    if name == "td3":
        critics = [ValueFunction(mdp.action_dim, mdp.state_dim, config["critic_hidden_size"])
                   for _ in range(2)]
        return TD3(
            **common, value_function_1=critics[0], value_function_2=critics[1],
            value_functions_optimiser=Adam(
                chain(*(critic.parameters() for critic in critics)), lr=config["critic_lr"]
            ),
        )

    critic = QuantileFunction(
        q_start=config["q_start"], q_end=config["q_end"],
        nr_quantiles=config["nr_quantiles"],
        action_dim=mdp.action_dim, state_dim=mdp.state_dim,
        hidden_size=config["critic_hidden_size"],
    )
    extra = (
        {"mse_bound": config["huber_loss_mse_bound"]}
        if name == "d4pg_qr"
        else {
            "mse_bound_decay": config["huber_loss_mse_bound_decay"],
            "loss_type": "AGQR" if config["use_approximate_loss"] else "GQR",
        }
    )
    return (D4PG_QR if name == "d4pg_qr" else D4PG_GQR)(
        **common, objective_function=_quantile_objective,
        quantile_function=critic,
        quantile_function_optimiser=Adam(critic.parameters(), lr=config["critic_lr"]),
        **extra,
    )

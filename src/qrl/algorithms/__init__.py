__all__ = [
    "MD4PG",
    "D4PG_QR",
    "DDPG",
    "PPO",
    "REINFORCE",
    "TD3",
    "VMPO"
]
from .d4pg_qr import D4PG_QR
from .ddpg import DDPG
from .md4pg import MD4PG
from .ppo import PPO
from .reinforce import REINFORCE
from .td3 import TD3
from .vmpo import VMPO
from ..functions import (
    Policy,
    GaussianNoiseScale,
    GaussianPolicy,
    DeterministicPolicy,
    ValueFunction,
    StateValueFunction,
    QuantileFunction,
)
import abc
from dataclasses import dataclass
from ..mdp import MDP
from torch.optim import Adam
from torch import tensor
from itertools import chain
import yaml

@dataclass
class Algorithm(abc.ABC):
    mdp: MDP
    policy: Policy

    @abc.abstractmethod
    def train(self, rounds: int) -> None:
        raise NotImplementedError

def get_algorithm(algorithm_name: str, mdp: MDP) -> Algorithm:
    name = algorithm_name.lower()
    assert name in [
        "md4pg",
        "d4pg_qr",
        "ddpg",
        "ppo",
        "reinforce",
        "td3",
        "vmpo",
    ]
    with open("config/algorithms.yaml") as file:
        config = yaml.safe_load(file)
        if name == "reinforce":
            policy = GaussianPolicy(
                action_dim=mdp.action_dim, 
                state_dim=mdp.state_dim, 
                hidden_size=config[name]["policy_hidden_size"], 
                action_lb=mdp.action_lb, 
                action_ub=mdp.action_ub, 
                min_std=config[name]["policy_min_std"],
                max_std=config[name]["policy_max_std"]
            )
            return REINFORCE(
                mdp=mdp,
                policy=policy,
                optimiser=Adam(policy.parameters(), lr=config[name]["lr"]),
                **config[name]["kwargs"]
            )
        elif name == "vmpo":
            policy = GaussianPolicy(
                action_dim=mdp.action_dim, 
                state_dim=mdp.state_dim, 
                hidden_size=config[name]["policy_hidden_size"], 
                action_lb=mdp.action_lb, 
                action_ub=mdp.action_ub, 
                min_std=config[name]["policy_min_std"],
                max_std=config[name]["policy_max_std"]
            )
            critic = StateValueFunction(
                mdp.state_dim,
                config[name]["critic_hidden_size"],
            )
            return VMPO(
                mdp=mdp,
                policy=policy,
                policy_optim=Adam(policy.parameters(), lr=config[name]["policy_lr"]),
                value_function=critic,
                value_function_optim=Adam(critic.parameters(), lr=config[name]["critic_lr"]),
                **config[name]["kwargs"]
            )
        elif name == "ppo":
            policy = GaussianPolicy(
                action_dim=mdp.action_dim, 
                state_dim=mdp.state_dim, 
                hidden_size=config[name]["policy_hidden_size"], 
                action_lb=mdp.action_lb, 
                action_ub=mdp.action_ub, 
                min_std=config[name]["policy_min_std"],
                max_std=config[name]["policy_max_std"]
            )
            critic = StateValueFunction(
                mdp.state_dim,
                config[name]["critic_hidden_size"],
            )
            return PPO(
                mdp=mdp,
                policy=policy,
                policy_optim=Adam(policy.parameters(), lr=config[name]["policy_lr"]),
                value_function=critic,
                value_function_optim=Adam(critic.parameters(), lr=config[name]["critic_lr"]),
                **config[name]["kwargs"]
            )
        elif name == "ddpg":
            policy = DeterministicPolicy(
                action_dim=mdp.action_dim,
                state_dim=mdp.state_dim,
                hidden_size=config[name]["policy_hidden_size"],
                action_lb=mdp.action_lb,
                action_ub=mdp.action_ub
            )
            critic = ValueFunction(
                action_dim=mdp.action_dim,
                state_dim=mdp.state_dim,
                hidden_size=config[name]["critic_hidden_size"]
            )
            return DDPG(
                mdp=mdp,
                policy=policy,
                value_function=critic,
                policy_optimiser=Adam(policy.parameters(), lr=config[name]["policy_lr"]),
                value_function_optimiser=Adam(critic.parameters(), lr=config[name]["critic_lr"]),
                exploration_noise=GaussianNoiseScale(
                    initial_value=tensor([config[name]["initial_exploration_noise"]]),
                    update_frequency=config[name]["noise_update_frequency"],
                    decay_factor=config[name]["noise_decay_factor"],
                    minimum_value=tensor([config[name]["minimum_exploration_noise"]])
                ),
                **config[name]["kwargs"]
            )
        elif name == "td3":
            policy = DeterministicPolicy(
                action_dim=mdp.action_dim,
                state_dim=mdp.state_dim,
                hidden_size=config[name]["policy_hidden_size"],
                action_lb=mdp.action_lb,
                action_ub=mdp.action_ub
            )
            critic1 = ValueFunction(
                action_dim=mdp.action_dim,
                state_dim=mdp.state_dim,
                hidden_size=config[name]["critic_hidden_size"]
            )
            critic2 = ValueFunction(
                action_dim=mdp.action_dim,
                state_dim=mdp.state_dim,
                hidden_size=config[name]["critic_hidden_size"]
            )
            return TD3(
                mdp=mdp,
                policy=policy,
                value_function_1=critic1,
                value_function_2=critic2,
                policy_optimiser=Adam(policy.parameters(), lr=config[name]["policy_lr"]),
                value_functions_optimiser=Adam(chain(critic1.parameters(), critic2.parameters()), lr=config[name]["critic_lr"]),
                exploration_noise=GaussianNoiseScale(
                    initial_value=tensor([config[name]["initial_exploration_noise"]]),
                    update_frequency=config[name]["noise_update_frequency"],
                    decay_factor=config[name]["noise_decay_factor"],
                    minimum_value=tensor([config[name]["minimum_exploration_noise"]])
                ),
                **config[name]["kwargs"]
            )
        elif name == "md4pg":
            raise NotImplementedError
        elif name == "d4pg_qr":
            raise NotImplementedError
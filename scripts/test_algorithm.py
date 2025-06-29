import torch
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from qrl.mdp import LQGMDP
from qrl.algorithms import get_algorithm
from qrl.functions import GaussianPolicy
torch.manual_seed(0)
rng = np.random.default_rng(0) 

def make_mdp(nr_paths: int, nr_steps: int) -> LQGMDP:
    noise_paths = torch.from_numpy(
        norm.ppf(LatinHypercube(nr_steps, scramble=False, rng=rng).random(nr_paths))
    ).float() * 0.15

    state_transform = {
        t: torch.tensor([[1.0, 1.0], [0.0, 1.0 - 0.005 * t]]).float()
        for t in range(nr_steps)
    }

    control_transform = {
        t: torch.tensor([[0.0], [1.0 - 0.01 * t]]).clamp(min=0.5).float()
        for t in range(nr_steps)
    }

    state_penalty = {
        t: torch.tensor([[1.0 + 0.05 * t, 0.0], [0.0, 0.1]]).float()
        for t in range(nr_steps)
    }

    control_penalty = {
        t: torch.tensor([[0.01 + 0.001 * t]]).float()
        for t in range(nr_steps)
    }

    return LQGMDP(
        action_lb=torch.tensor([-100.0]).float(),
        action_ub=torch.tensor([100.0]).float(),
        noise_paths=noise_paths,
        state_transform=state_transform,
        control_transform=control_transform,
        state_penalty=state_penalty,
        final_state_penalty=torch.tensor([[10.0, 0.0], [0.0, 2.0]]).float(),
        control_penalty=control_penalty,
        initial_state=torch.tensor([2.0, -1.0]).float(),
    )


def main():
    algorithm_name = "TD3"
    nr_paths = 1000
    nr_steps = 50
    training_rounds = 250
    mdp = make_mdp(nr_paths, nr_steps)
    benchmark_actions, benchmark_rewards = mdp.solve()

    algorithm = get_algorithm(algorithm_name, mdp)

    algorithm.train(training_rounds)

    if isinstance(algorithm.policy, GaussianPolicy):
        algorithm.policy.deterministic(True)
    actions, rewards = algorithm.mdp.evaluate(algorithm.policy)
    if isinstance(algorithm.policy, GaussianPolicy):
        algorithm.policy.deterministic(False)

    cumulative_rewards = rewards.sum(dim=1).numpy()
    benchmark_cumulative_rewards = benchmark_rewards.sum(dim=1).numpy()

    plt.title("Cumulative Rewards Histogram")
    plt.hist(cumulative_rewards, color="blue", alpha=0.5, label="RL")
    plt.hist(benchmark_cumulative_rewards, color="orange", alpha=0.5, label="Benchmark")
    plt.axvline(cumulative_rewards.mean(), c="blue", linestyle="--")
    plt.axvline(benchmark_cumulative_rewards.mean(), c="orange", linestyle="--")
    plt.legend()
    plt.show()

    plt.title("Expected Reward Per Timestep")
    plt.plot(rewards.mean(dim=0).numpy(), label="RL", c="blue")
    plt.axhline(rewards.mean(dim=0).numpy().mean(), linestyle="--", c="blue")
    plt.plot(benchmark_rewards.mean(dim=0).numpy(), label="Benchmark", c="orange")
    plt.axhline(benchmark_rewards.mean(dim=0).numpy().mean(), linestyle="--", c="orange")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
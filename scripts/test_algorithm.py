import torch
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from qrl.mdp import LQGMDP
from qrl.algorithms import get_algorithm
from qrl.functions import GaussianPolicy
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid")
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
    save_fig = True
    algorithm_name = "D4PG_QR"
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

    plt.figure(figsize=(10, 6))
    sns.histplot(cumulative_rewards, color="blue", label="RL", kde=False, stat="density", bins=30, alpha=0.6)
    sns.histplot(benchmark_cumulative_rewards, color="orange", label="Benchmark", kde=False, stat="density", bins=30, alpha=0.6)

    plt.axvline(cumulative_rewards.mean(), color="blue", linestyle="--", label="RL Mean")
    plt.axvline(benchmark_cumulative_rewards.mean(), color="orange", linestyle="--", label="Benchmark Mean")

    plt.title("Cumulative Rewards Distribution", fontsize=18)
    plt.xlabel("Cumulative Reward", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if not save_fig:
        plt.show()
    else:
        plt.savefig(f"plots/{algorithm_name}_{nr_steps}_{training_rounds}_hist.png", format="png")
        plt.close()

    plt.figure(figsize=(10, 6))
    mean_rl = rewards.mean(dim=0).numpy()
    mean_benchmark = benchmark_rewards.mean(dim=0).numpy()

    sns.lineplot(x=range(len(mean_rl)), y=mean_rl, label="RL", color="C0", linewidth=2)
    plt.axhline(mean_rl.mean(), linestyle="--", color="C0", linewidth=1.5, label="RL Mean")

    sns.lineplot(x=range(len(mean_benchmark)), y=mean_benchmark, label="Benchmark", color="C1", linewidth=2)
    plt.axhline(mean_benchmark.mean(), linestyle="--", color="C1", linewidth=1.5, label="Benchmark Mean")


    plt.title("Expected Reward per Timestep", fontsize=18)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Expected Reward", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if not save_fig:
        plt.show()
    else:
        plt.savefig(f"plots/{algorithm_name}_{nr_steps}_{training_rounds}_per_time.png", format="png")
        plt.close()


if __name__ == "__main__":
    main()
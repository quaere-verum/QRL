import torch
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from qrl.mdp import LQGMDP, TradeExecutionMDP
from qrl.algorithms import get_algorithm
from qrl.functions import GaussianPolicy
import seaborn as sns
sns.set_theme(context="paper", style="whitegrid")
torch.manual_seed(0)
rng = np.random.default_rng(0) 

def make_lqg_mdp(nr_paths: int, nr_steps: int) -> LQGMDP:
    noise_paths = torch.from_numpy(
        norm.ppf(LatinHypercube(nr_steps, scramble=False, rng=rng).random(nr_paths))
    ).float()[:, :, torch.newaxis] * 0.25

    state_transform = {
        t: torch.tensor([
            [1.02, 1.1],
            [0.0, 1.005 + 0.0015 * t]
        ]).float()
        for t in range(nr_steps)
    }

    control_transform = {
        t: torch.tensor([[0.0], [1.0 - 0.025 * t]]).clamp(min=0.5).float()
        for t in range(nr_steps)
    }

    state_penalty = {
        t: torch.tensor([
            [1.0 + 0.05 * t, 0.25],
            [0.25, 0.2 + 0.02 * t]
        ]).float()
        for t in range(nr_steps)
    }

    control_penalty = {
        t: torch.tensor([[0.001 if t < nr_steps // 2 else 0.05 + 0.005 * (t - nr_steps//2)]]).float()
        for t in range(nr_steps)
    }

    return LQGMDP(
        action_lb=torch.tensor([-100.0]).float(),
        action_ub=torch.tensor([100.0]).float(),
        noise_paths=noise_paths,
        state_transform=state_transform,
        control_transform=control_transform,
        state_penalty=state_penalty,
        final_state_penalty=torch.tensor([[15.0, 5.0], [5.0, 10.0]]).float(),
        control_penalty=control_penalty,
        initial_state=torch.tensor([2.0, -1.0]).float(),
    )

def make_trade_mdp(nr_paths: int, nr_steps: int) -> TradeExecutionMDP:
    noise_paths = torch.from_numpy(
        norm.ppf(LatinHypercube(nr_steps, scramble=False, rng=rng).random(nr_paths))
    ).float()

    initial_inventory = 1_000.0

    return TradeExecutionMDP(
        action_lb=torch.tensor([-initial_inventory / nr_steps * 2]).float(),
        action_ub=torch.tensor([initial_inventory / nr_steps * 2]).float(),
        noise_paths=noise_paths,
        initial_inventory=initial_inventory,
        initial_price=100.0,
        sigma=0.5,
        eta=1e-3,
        lam=1e-6,
        gamma=1e-6,
        kappa=1e-4,
        impact_fn="power",
        impact_power=1.8,
        impact_eps=1e-2,
        temp_cost_fn="power",
        temp_power_delta=0.5,
    )


def main():
    save_fig = True
    algorithm_name = "MD4PG"
    nr_paths = 1000
    nr_steps = 10
    mdp = make_trade_mdp(nr_paths, nr_steps)
    benchmark_actions, benchmark_rewards = mdp.solve()
    for algorithm_name, training_rounds in [
        # ("REINFORCE", 500),
        # ("PPO", 3000),
        # ("VMPO", 4500),
        # ("TD3", 500),
        # ("D4PG_QR", 100),
        ("D4PG_GQR", 50),
    ]:

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

        plt.title(f"{algorithm_name} - Cumulative Rewards Distribution", fontsize=18)
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
        print("RL=", mean_rl.mean())
        print("Benchmark=", mean_benchmark.mean())


        plt.title(f"{algorithm_name} - Expected Reward per Timestep", fontsize=18)
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
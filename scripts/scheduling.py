"""Train PPO/VMPO on two small scheduling benchmarks with learning diagnostics.

Stage 1: one server, log-uniform processing times, exact SPT optimum.
Stage 2: two class-specialised servers, exact offline assignment lower bound.

All Monte Carlo scenarios are fixed across training rounds. The test scenarios
are independent and shared across policy checkpoints and greedy comparisons.

Examples:
    uv run scripts/scheduling.py --stage 1 --algorithm ppo --n-rounds 500
    uv run scripts/scheduling.py --stage 2 --algorithm vmpo --n-paths 10000 \\
        --n-test-paths 10000 --n-rounds 1000 --eval-every 50
"""

import argparse
import csv
from contextlib import contextmanager
from math import log, sqrt
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from qrl.algorithms import get_algorithm
from qrl.mdp import SchedulingMDP


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--algorithm", choices=("ppo", "vmpo"), default="ppo")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-paths", type=int, default=2048)
    parser.add_argument("--n-test-paths", type=int, default=2048)
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--test-distribution", choices=("loguniform", "lognormal"),
        default="loguniform",
    )
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.n_jobs, args.n_paths, args.n_test_paths, args.eval_every) < 1:
        parser.error("jobs, paths and eval-every must be positive")
    if args.n_rounds < 0:
        parser.error("n-rounds must be nonnegative")
    if args.output_prefix is None:
        args.output_prefix = Path("results") / f"scheduling_stage{args.stage}_{args.algorithm}_seed{args.seed}"
    return args


def draw_workloads(
    n_paths: int, n_jobs: int, rng: torch.Generator, distribution: str,
) -> torch.Tensor:
    if distribution == "loguniform":
        u = torch.rand((n_paths, n_jobs), generator=rng)
        return torch.exp(log(0.5) + u * log(8.0 / 0.5))
    if distribution == "lognormal":
        return torch.exp(log(2.0) + 0.65 * torch.randn((n_paths, n_jobs), generator=rng))
    raise ValueError(distribution)


def make_mdp(
    stage: int, n_paths: int, n_jobs: int, seed: int,
    distribution: str = "loguniform",
) -> SchedulingMDP:
    rng = torch.Generator().manual_seed(seed)
    workloads = draw_workloads(n_paths, n_jobs, rng, distribution)

    if stage == 1:
        service_times = workloads.unsqueeze(-1)
        features = None
    else:
        # Classes specialise in server 0, server 1, or neither.
        speeds = torch.tensor([[2.0, 0.7], [0.7, 2.0], [1.0, 1.0]])
        classes = torch.randint(0, len(speeds), (n_paths, n_jobs), generator=rng)
        service_times = workloads.unsqueeze(-1) / speeds[classes]
        features = torch.nn.functional.one_hot(classes, len(speeds)).float()

    return SchedulingMDP(
        action_lb=torch.tensor([0]),
        action_ub=torch.tensor([n_jobs * service_times.shape[-1]]),
        arrival_times=torch.zeros((n_paths, n_jobs)),
        service_times=service_times,
        runtime_estimates=service_times.clone(),
        job_features=features,
        n_candidates=n_jobs,
        history_length=0,
    )


def optimal_costs(mdp: SchedulingMDP) -> np.ndarray:
    """Offline optimum with unrestricted server idling (Stage 2 lower bound)."""
    times = mdp.service_times.cpu().numpy()
    n_jobs, n_servers = times.shape[1:]
    if n_servers == 1:
        return np.cumsum(np.sort(times[:, :, 0], axis=1), axis=1).sum(axis=1) / n_jobs

    weights = np.arange(1, n_jobs + 1, dtype=np.float64)
    costs = np.empty(times.shape[0], dtype=np.float64)
    for b, durations in enumerate(times):
        matrix = (durations[:, :, None] * weights).reshape(n_jobs, n_servers * n_jobs)
        rows, cols = linear_sum_assignment(matrix)
        costs[b] = matrix[rows, cols].sum() / n_jobs
    return costs


class GreedyPolicy:
    """Choose the legal job-server assignment with shortest estimated duration."""

    def __init__(self, mdp: SchedulingMDP):
        self.n_candidates = mdp.n_candidates
        self.n_servers = mdp.n_servers
        self.n_actions = mdp.n_actions
        self.feature_dim = mdp.job_features.shape[-1]

    def act(self, state: torch.Tensor) -> torch.Tensor:
        width = 2 + self.n_servers + self.feature_dim
        candidates = state[:, 5:5 + self.n_candidates * width].reshape(
            -1, self.n_candidates, width
        )
        estimates = candidates[:, :, 2:2 + self.n_servers].reshape(-1, self.n_actions)
        mask = state[:, -self.n_actions:] > 0.5
        return estimates.masked_fill(~mask, torch.inf).argmin(dim=1, keepdim=True).long()


@contextmanager
def deterministic_mode(policy):
    """Restore the policy's previous action-selection mode, even on failure."""
    if not hasattr(policy, "deterministic"):
        yield
        return
    old_mode = getattr(policy, "_deterministic_mode", False)
    policy.deterministic(True)
    try:
        yield
    finally:
        policy.deterministic(old_mode)


def evaluate(mdp: SchedulingMDP, policy, *, entropy: bool = False):
    """Deterministic policy cost and entropy on the states that it visits."""
    entropy_sum = 0.0
    choice_count = 0
    total_decisions = mdp.nr_paths * mdp.nr_steps
    with deterministic_mode(policy), torch.no_grad():
        state = mdp.reset()
        rewards = torch.zeros(mdp.nr_paths, dtype=torch.float64)
        for _ in range(mdp.nr_steps):
            if entropy:
                legal = (state[:, -mdp.n_actions:] > 0.5).sum(dim=1)
                chosen = legal > 1
                if chosen.any():
                    h = policy.get_dist(state).entropy()
                    entropy_sum += (h[chosen] / legal[chosen].float().log()).sum().item()
                    choice_count += int(chosen.sum().item())
            action = policy.act(state)
            state, reward = mdp.step(action)
            rewards += reward.double()
        costs = mdp.mean_sojourn()
        torch.testing.assert_close(-rewards, costs, rtol=1e-5, atol=1e-5)
    result = dict(costs=costs.cpu().numpy().copy())
    if entropy:
        result["entropy"] = entropy_sum / choice_count if choice_count else float("nan")
        result["choice_fraction"] = choice_count / total_decisions
    return result


def critic_explained_variance(mdp: SchedulingMDP, policy, critic, seed: int) -> float:
    """Evaluate V(s) against realised *stochastic-policy* returns.

    Preserve the training RNG state so checkpoint diagnostics do not alter the
    subsequent PPO/VMPO training trajectory.
    """
    old_mode = getattr(policy, "_deterministic_mode", False)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        policy.deterministic(False)
        try:
            state = mdp.reset()
            rewards = torch.empty((mdp.nr_paths, mdp.nr_steps))
            values = torch.empty_like(rewards)
            for t in range(mdp.nr_steps):
                values[:, t] = critic.value(state).flatten()
                action = policy.act(state)
                state, rewards[:, t] = mdp.step(action)
        finally:
            policy.deterministic(old_mode)

    returns = torch.flip(torch.cumsum(torch.flip(rewards, (1,)), dim=1), (1,))
    target = returns.double().reshape(-1)
    residual = (returns - values).double().reshape(-1)
    variance = torch.var(target, unbiased=False).item()
    if variance < 1e-12:
        return float("nan")
    return 1.0 - torch.var(residual, unbiased=False).item() / variance


def parameter_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        norms = [p.detach().double().norm() for p in model.parameters()]
        if not norms or not all(torch.isfinite(x).item() for x in norms):
            raise FloatingPointError("Non-finite neural-network parameters")
        return torch.stack(norms).norm().item()


@contextmanager
def gradient_monitor(algorithm):
    """Record PRE-clipping gradient norms without editing QRL's training code.

    Both PPO and VMPO call torch.nn.utils.clip_grad_norm_. Its return value
    is the norm *before* clipping. A temporary wrapper records that value and
    delegates to the original function without changing its behaviour.
    Intended for the single-threaded diagnostic script, not concurrent runs.
    """
    policy_ids = {id(p) for p in algorithm.policy.parameters()}
    critic_ids = {id(p) for p in algorithm.value_function.parameters()}
    record = {"policy": [], "critic": []}
    clipped = {"policy": 0, "critic": 0}
    original = torch.nn.utils.clip_grad_norm_

    def observed(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        raw = original(parameters, max_norm, *args, **kwargs)
        which = "policy" if any(id(p) in policy_ids for p in parameters) else "critic"
        if not any(id(p) in (policy_ids if which == "policy" else critic_ids) for p in parameters):
            return raw
        value = float(raw.item())
        record[which].append(value)
        clipped[which] += int(value > float(max_norm))
        return raw

    with patch.object(torch.nn.utils, "clip_grad_norm_", observed):
        yield record, clipped


def gradient_summary(record: dict, clipped: dict) -> dict:
    result = {}
    for name in ("policy", "critic"):
        values = np.asarray(record[name], dtype=float)
        result[f"{name}_grad_norm"] = float(values.mean()) if len(values) else float("nan")
        result[f"{name}_grad_max"] = float(values.max()) if len(values) else float("nan")
        result[f"{name}_clip_fraction"] = clipped[name] / len(values) if len(values) else float("nan")
        if len(values) and not np.isfinite(values).all():
            raise FloatingPointError(f"Non-finite {name} gradient norms")
    return result


def paired_se(differences: np.ndarray) -> float:
    return float(np.std(differences, ddof=1) / sqrt(len(differences))) if len(differences) > 1 else float("nan")


def plot_history(rows: list[dict], prefix: Path):
    rounds = np.array([r["round"] for r in rows])

    def save_plot(name, title, ylabel, series):
        plt.figure(figsize=(8, 4.5))
        for label, values, linestyle in series:
            plt.plot(rounds, values, linestyle, label=label)
        plt.title(title)
        plt.xlabel("Training rounds")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + f"_{name}.png"), dpi=150)
        plt.close()

    def col(name):
        return [r[name] for r in rows]

    save_plot("cost", "Deterministic scheduling performance", "Mean sojourn time", [
        ("Training", col("train_cost"), "-"),
        ("Test", col("test_cost"), "-"),
        ("Greedy", col("greedy_cost"), "--"),
        ("Offline bound", col("oracle_cost"), ":"),
    ])
    save_plot("entropy", "Policy entropy on states with multiple legal actions", "Normalised entropy", [
        ("Test entropy", col("entropy"), "-"),
    ])
    save_plot("critic", "Value-function diagnostics", "Explained variance", [
        ("Test critic EV", col("critic_ev"), "-"),
    ])
    save_plot("gradients", "Pre-clipping gradient norms", "Mean gradient L2 norm", [
        ("Policy", col("policy_grad_norm"), "-"),
        ("Critic", col("critic_grad_norm"), "-"),
    ])


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)

    train = make_mdp(args.stage, args.n_paths, args.n_jobs, args.seed)
    test = make_mdp(args.stage, args.n_test_paths, args.n_jobs, args.seed + 1,
                    args.test_distribution)
    greedy = evaluate(test, GreedyPolicy(test))["costs"]
    oracle = optimal_costs(test)
    if np.any(greedy + 1e-5 < oracle):
        raise AssertionError("Greedy beat the offline optimum")
    if args.stage == 1:
        np.testing.assert_allclose(greedy, oracle, atol=1e-5, rtol=1e-6)

    algorithm = get_algorithm(
        args.algorithm, train, policy_type="categorical", mask_from_state=True,
        gamma=1.0, gae_lambda=1.0,
    )

    print(f"Stage {args.stage}: {args.n_jobs} jobs, {train.n_servers} server(s)")
    print(f"Fixed training/test scenarios: {args.n_paths:,} / {args.n_test_paths:,}")
    print(f"Test distribution: {args.test_distribution}")
    print(f"Greedy mean: {greedy.mean():.4f}; offline bound: {oracle.mean():.4f}")
    print("Round   Train     Test     DeltaGreedy +/- SE    Excess%   Hnorm    CriticEV   Pgrad   Cgrad   Pclip Cclip")

    rows = []
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")

    def checkpoint(round_number: int, stats: dict):
        train_result = evaluate(train, algorithm.policy)
        test_result = evaluate(test, algorithm.policy, entropy=True)
        cost = test_result["costs"]
        diff = cost - greedy
        row = dict(
            round=round_number,
            train_cost=float(train_result["costs"].mean()),
            test_cost=float(cost.mean()),
            greedy_cost=float(greedy.mean()),
            oracle_cost=float(oracle.mean()),
            delta_greedy=float(diff.mean()),
            paired_se=paired_se(diff),
            delta_oracle=float((cost - oracle).mean()),
            excess_pct=100.0 * (cost.mean() / oracle.mean() - 1.0),
            entropy=float(test_result["entropy"]),
            choice_fraction=float(test_result["choice_fraction"]),
            critic_ev=critic_explained_variance(test, algorithm.policy,
                                                algorithm.value_function,
                                                seed=args.seed + 100003),
            policy_param_norm=parameter_norm(algorithm.policy),
            critic_param_norm=parameter_norm(algorithm.value_function),
            **stats,
        )
        essential = (row["train_cost"], row["test_cost"],
                     row["policy_param_norm"], row["critic_param_norm"])
        if not np.isfinite(essential).all():
            raise FloatingPointError(f"Non-finite parameters or costs at round {round_number}")
        # Entropy and EV can be undefined when there is no action choice or
        # essentially no variation in realised returns, respectively.
        if row["choice_fraction"] > 0 and not np.isfinite(row["entropy"]):
            raise FloatingPointError(f"Non-finite policy entropy at round {round_number}")
        rows.append(row)
        # Save at every checkpoint, so an interrupted run retains its history.
        with csv_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{round_number:5d} {row['train_cost']:8.4f} {row['test_cost']:8.4f} "
              f"{row['delta_greedy']:+9.4f} +/-{row['paired_se']:.4f} "
              f"{row['excess_pct']:8.2f} {row['entropy']:7.3f} "
              f"{row['critic_ev']:9.3f} {row['policy_grad_norm']:7.3f} "
              f"{row['critic_grad_norm']:7.3f} "
              f"{row['policy_clip_fraction']:5.2f} {row['critic_clip_fraction']:5.2f}", flush=True)

    empty = gradient_summary({"policy": [], "critic": []}, {"policy": 0, "critic": 0})
    checkpoint(0, empty)
    for start in range(0, args.n_rounds, args.eval_every):
        n = min(args.eval_every, args.n_rounds - start)
        with gradient_monitor(algorithm) as (records, clipped):
            algorithm.train(n)
        checkpoint(start + n, gradient_summary(records, clipped))

    if not args.no_plots:
        plot_history(rows, args.output_prefix)
    print(f"Diagnostics saved to {csv_path}")
    if args.stage == 2:
        print("Note: the offline optimum can require idling, which this MDP forbids.")


if __name__ == "__main__":
    main()

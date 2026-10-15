"""Small PPO smoke test for QRL's SchedulingMDP.

Run from the QRL repository after installing it (e.g. ``poetry run python
scripts/train_scheduling.py``). This is an exactly solvable scheduling problem:

* Four jobs arrive at time zero on one server.
* Service times are a random permutation of [1, 2, 4, 8].
* The service-time estimates are exact.
* Every episode contains four dispatch decisions.

The optimal shortest-processing-time-first mean sojourn time is 6.5;
the expectation under a uniformly random order is 9.375.
"""

import torch

from qrl.algorithms import get_algorithm, Policy
from qrl.mdp import SchedulingMDP
import argparse


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=128)
    parser.add_argument("--n-test-paths", type=int, default=128)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--algorithm", type=str, default="ppo")
    return parser.parse_args(argv)
    

def make_mdp(n_paths: int, seed: int) -> SchedulingMDP:
    rng = torch.Generator().manual_seed(seed)
    durations = torch.tensor([1.0, 2.0, 4.0, 8.0])
    permutation = torch.rand(n_paths, len(durations), generator=rng).argsort(dim=1)
    service_times = durations[permutation].unsqueeze(-1)  # (B, 4 jobs, 1 server)

    return SchedulingMDP(
        action_lb=torch.tensor([0]),
        action_ub=torch.tensor([len(durations)]),
        arrival_times=torch.zeros(n_paths, len(durations)),
        service_times=service_times,
        runtime_estimates=service_times.clone(),
        n_candidates=len(durations),
        history_length=0,
    )


def evaluate(mdp: SchedulingMDP, policy: Policy) -> float:
    policy.deterministic(True)
    try:
        with torch.no_grad():
            _, rewards = mdp.evaluate(policy)
            sojourn = mdp.mean_sojourn()
            # The integrated holding cost must equal the realised sojourn cost.
            torch.testing.assert_close(-rewards.sum(dim=1).double(), sojourn)
            return sojourn.mean().item()
    finally:
        policy.deterministic(False)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    torch.manual_seed(args.seed)

    train_mdp = make_mdp(args.n_paths, seed=args.seed)
    test_mdp = make_mdp(args.n_test_paths, seed=args.seed + 1)

    algorithm = get_algorithm(
        args.algorithm,
        train_mdp,
        policy_type="categorical",
        mask_from_state=True,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected_random = 9.375
    optimal_spt = 6.5
    before = evaluate(test_mdp, algorithm.policy)
    print(f"SPT optimum:             {optimal_spt:.3f}")
    print(f"Random-order expectation: {expected_random:.3f}")
    print(f"Before PPO training:      {before:.3f}")

    algorithm.train(args.n_rounds)

    after = evaluate(test_mdp, algorithm.policy)
    print(f"After PPO training:       {after:.3f}")
    print(f"Gap to optimum:           {after - optimal_spt:.3f}")


if __name__ == "__main__":
    main()

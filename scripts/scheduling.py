"""Train PPO/VMPO on two small scheduling benchmarks with learning diagnostics.

Stage 1: one server, log-uniform processing times, exact SPT optimum.
Stage 2: two class-specialised servers, exact offline assignment lower bound.

All Monte Carlo scenarios are fixed across training rounds. Checkpoints use
independent validation scenarios (the historical CSV column is ``test_cost``).
Use --final-test only once the training setup is finalised; it evaluates the
final policies on a third, untouched collection, without checkpoint selection.

Examples:
    uv run scripts/scheduling.py --stage 1 --algorithm ppo --n-rounds 500
    uv run scripts/scheduling.py --stage 2 --algorithm vmpo --n-paths 10000 \\
        --n-test-paths 10000 --n-rounds 1000 --eval-every 50
    uv run scripts/scheduling.py --stage 2 --experiment all --bc-epochs 25
    # Repeat with --seed 124/125 and --scenario-seed 123 for common workloads.
    uv run scripts/scheduling.py --experiment optimisation --n-paths 10000 \\
        --n-test-paths 10000 --n-rounds 1000 --seed 123 --scenario-seed 123 \\
        --final-test-seed 1000125 --run-name optimisation-123
    uv run scripts/scheduling.py --experiment oracle \\
        --comparison-suite results/20261015/optimisation-123 --n-paths 10000 \\
        --n-test-paths 10000 --n-rounds 1000 --bc-epochs 25 --seed 123 \\
        --scenario-seed 123 --final-test-seed 1000125 --run-name oracle-123

Use --experiment bc, ppo, or bc-ppo for individual arms. The bc-ppo arm also
reports its BC pretraining. --n-test-paths sizes both validation and final test.
CSV rows identify experiment, split, BC epoch and PPO round; test_cost retains
its historical name and refers to the indicated split. Accuracies/agreements
are fractions. BC cross-entropies include both all-state and choice-only means.
train_seconds, actor_updates and training_examples are cumulative and include
BC pretraining for bc-ppo; bc_seconds/ppo_seconds separate optimisation effort.
demonstration_seconds is shared collection time, diagnostic_seconds is the
current checkpoint's evaluation time. Final tests evaluate final weights only.
Each invocation writes fixed filenames under results/YYYYMMDD/<run-name>/.
Use --run-name to choose the directory name (default: <experiment>-<seed>).
Existing settings.json files are never overwritten; choose a new run name.
All checkpoints now save weights.pt and aligned deterministic/stochastic costs.npz
under <method>/<split>/<epoch-or-round>/. Stochastic evaluation uses one trajectory
per path and a separate fixed RNG, without perturbing PPO training.
The optimisation suite runs exactly the five specified scratch budgets and two
BC-PPO variants. It uses 1,000 critic warm-up steps by default, resets critic Adam
state, and tests validation-selected final configurations only after all training.
ppo_actor_updates/ppo_critic_updates and ppo_sampled_transitions exclude BC/warm-up;
their effort is reported separately. wall_seconds includes diagnostics and I/O.
The oracle experiment trains only optimal-action imitation and reuses completed
PPO controls. Its exact non-idling DP is distinct from the old idling lower bound.
It saves equivalent optimal-action sets and pathwise Bellman regrets; the latter
sum to the realised gap to the true optimum. The final BC epoch is fixed in advance.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from importlib.resources import files
from math import log, sqrt
from pathlib import Path
from time import perf_counter, process_time, sleep
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from qrl.algorithms import get_algorithm
from qrl.mdp import SchedulingMDP


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--algorithm", choices=("ppo", "vmpo"), default="ppo")
    parser.add_argument("--experiment", choices=("bc", "ppo", "bc-ppo", "all", "optimisation", "oracle"),
                        help="Stage 2 comparison; omit to retain the original PPO/VMPO run")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-paths", type=int, default=2048)
    parser.add_argument("--n-test-paths", type=int, default=2048)
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scenario-seed", type=int, default=None,
                        help="Workload seed (defaults to --seed); splits use seed, seed+1, seed+2")
    parser.add_argument("--final-test-seed", type=int, default=None,
                        help="Reserved test workload seed (defaults to scenario-seed + 2)")
    parser.add_argument("--bc-epochs", type=int, default=25)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-updates", type=int, default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=0,
                        help="Critic-only MC updates before BC-initialised PPO; fresh Adam state afterward")
    parser.add_argument("--critic-warmup-batch-size", type=int, default=512)
    parser.add_argument("--suite-workers", type=int, choices=(1, 2), default=2,
                        help="Concurrent processes for --experiment optimisation")
    parser.add_argument("--comparison-suite", type=Path,
                        help="Completed optimisation suite to reuse for --experiment oracle")
    parser.add_argument("--final-test", action="store_true",
                        help="Evaluate final policies on untouched scenarios after all training")
    parser.add_argument(
        "--test-distribution", choices=("loguniform", "lognormal"),
        default="loguniform",
    )
    parser.add_argument("--run-name", default=None, help="Directory name within results/YYYYMMDD/")
    parser.add_argument("--output-date", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="Legacy option: parent selects results root, basename selects run name")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.output_date is not None:
        try:
            if datetime.strptime(args.output_date, "%Y%m%d").strftime("%Y%m%d") != args.output_date:
                raise ValueError
        except ValueError:
            parser.error("output-date must be YYYYMMDD")
    if min(args.n_jobs, args.n_paths, args.n_test_paths, args.eval_every) < 1:
        parser.error("jobs, paths and eval-every must be positive")
    if args.n_rounds < 0:
        parser.error("n-rounds must be nonnegative")
    if args.bc_epochs < 0 or args.bc_batch_size < 1 or not np.isfinite(args.bc_lr) or args.bc_lr <= 0:
        parser.error("bc-epochs must be nonnegative; bc-lr and bc-batch-size must be positive")
    if any(x is not None and x < 1 for x in (args.ppo_batch_size, args.ppo_updates)):
        parser.error("PPO batch size and updates must be positive")
    if args.critic_warmup_steps < 0 or args.critic_warmup_batch_size < 1:
        parser.error("warmup steps must be nonnegative and batch size positive")
    if args.critic_warmup_steps and args.experiment not in ("bc-ppo", "all", "optimisation"):
        parser.error("critic warm-up requires --experiment bc-ppo, all or optimisation")
    if args.experiment == "optimisation" and (args.ppo_batch_size or args.ppo_updates or args.final_test):
        parser.error("the optimisation suite fixes its grid and performs selected final testing automatically")
    if args.experiment == "oracle" and (args.comparison_suite is None or args.n_jobs != 8):
        parser.error("oracle comparison requires eight jobs and --comparison-suite")
    if args.experiment and (args.stage != 2 or args.algorithm != "ppo"):
        parser.error("--experiment requires --stage 2 and --algorithm ppo")
    if args.scenario_seed is None:
        args.scenario_seed = args.seed
    if args.final_test_seed is None:
        args.final_test_seed = args.scenario_seed + 2
    if args.final_test_seed in (args.scenario_seed, args.scenario_seed + 1):
        parser.error("final-test-seed must differ from training and validation seeds")
    if args.run_name is not None and args.output_prefix is not None:
        parser.error("use either --run-name or --output-prefix")
    if args.run_name is None:
        args.run_name = (args.output_prefix.name if args.output_prefix else
                         f"{args.experiment or args.algorithm}-{args.seed}")
    if (not args.run_name or args.run_name in (".", "..")
            or any(char in args.run_name for char in '<>:"/\\|?*')
            or args.run_name.endswith((".", " "))):
        parser.error("run-name must be a single valid directory name")
    return args


def prepare_output(args):
    """Reserve one run and save resolved settings before any expensive work."""
    now = datetime.now().astimezone()
    root = args.output_prefix.parent if args.output_prefix else Path("results")
    directory = root / (args.output_date or now.strftime("%Y%m%d")) / args.run_name
    directory.mkdir(parents=True, exist_ok=True)
    config_path = files("qrl.config").joinpath(f"{args.algorithm}.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["kwargs"].update(gamma=1.0, gae_lambda=1.0)
    if args.ppo_batch_size is not None:
        config["kwargs"]["batch_size"] = args.ppo_batch_size
    if args.ppo_updates is not None:
        config["kwargs"]["update_per_rollout"] = args.ppo_updates
    settings = dict(
        started_at=now.isoformat(),
        cli_arguments=sys.argv[1:], script=sys.argv[0],
        settings={key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()},
        algorithm_config=config,
        scenario_seeds=dict(training=args.scenario_seed, validation=args.scenario_seed + 1,
                            final_test=args.final_test_seed),
        runtime=dict(python=sys.version, torch=torch.__version__,
                     torch_threads=torch.get_num_threads(),
                     omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                     mkl_num_threads=os.environ.get("MKL_NUM_THREADS")),
    )
    # Exclusive creation also protects concurrent launches with the same name.
    with (directory / "settings.json").open("x", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")
    print(f"Run directory: {directory.resolve()}", flush=True)
    return directory


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


class ExactScheduler:
    """Exact non-idling DP for the fully observed, all-at-zero two-server case.

    A ternary digit records unassigned/server-0/server-1 for each job. With no
    voluntary idling, each server's availability is the sum of its assigned
    durations. Only the least-loaded server may dispatch (both on an exact tie).
    V records the minimum SUM of completion times of jobs still unassigned;
    completed/running jobs have already contributed their fixed completion time.
    Complexity O(B*N*3**N), memory O(B*3**N), rather than B*N! rollouts.
    """
    tolerance = 1e-9  # Mean-sojourn units, for equivalent actions only.

    def __init__(self, mdp, description="Exact non-idling oracle"):
        if (mdp.n_servers != 2 or mdp.nr_steps > 8 or mdp.n_candidates != mdp.nr_steps
                or torch.any(mdp.arrival_times != 0)
                or not torch.equal(mdp.service_times, mdp.runtime_estimates.double())):
            raise ValueError("Exact oracle requires <=8 jobs, two servers, all arrivals zero, "
                             "all jobs visible and exact runtime estimates")
        self.mdp, self.n = mdp, mdp.nr_steps
        self.times = mdp.service_times.cpu().numpy()
        exponents = np.frexp(self.times)[1]
        # <=36 duration terms in the objective: 24+23+6 <= 53 binary digits.
        if np.ptp(exponents) > 23 or self.times.max() > 1e4:
            raise ValueError("Duration range exceeds the exact-oracle numeric support")
        self.rows = np.arange(mdp.nr_paths)
        self.powers = 3 ** np.arange(self.n, dtype=np.int64)
        self.digits = (np.arange(3**self.n)[:, None] // self.powers) % 3
        self.masks = np.stack([((self.digits == s+1) * (2**np.arange(self.n))).sum(1)
                               for s in range(2)])
        self.loads = np.zeros((2, 2**self.n, mdp.nr_paths), dtype=np.float64)
        for mask in range(1, 2**self.n):
            bit = mask & -mask
            job = bit.bit_length()-1
            self.loads[:, mask] = self.loads[:, mask ^ bit] + self.times[:, job, :].T
        self.values = np.zeros((3**self.n, mdp.nr_paths), dtype=np.float64)
        started = perf_counter()
        for code in tqdm(range(3**self.n-1, -1, -1), desc=description, mininterval=1):
            remaining = np.flatnonzero(self.digits[code] == 0)
            if not len(remaining):
                continue
            loads = self.loads[np.arange(2), self.masks[:, code]]
            best = np.full(mdp.nr_paths, np.inf)
            for server in range(2):
                idle = loads[server] <= loads[1-server]
                for job in remaining:
                    value = (loads[server] + self.times[:, job, server]
                             + self.values[code + (server+1)*self.powers[job]])
                    np.minimum(best, np.where(idle, value, np.inf), out=best)
            self.values[code] = best
        self.costs = self.values[0] / self.n
        self.seconds = perf_counter()-started
        print(f"{description}: {mdp.nr_paths:,} paths, {3**self.n:,} states/path, "
              f"{self.values.nbytes/2**20:.1f} MiB, {self.seconds:.2f}s", flush=True)

    def action_values(self, codes):
        candidates, legal = self.mdp._candidates_and_mask()
        candidates, legal = candidates.numpy(), legal.numpy()
        q = np.full(legal.shape, np.inf)
        loads = np.stack([self.loads[s, self.masks[s, codes], self.rows] for s in range(2)])
        for slot in range(self.n):
            jobs = candidates[:, slot]
            for server in range(2):
                a = 2*slot + server
                rows = self.rows[legal[:, a]]
                js = jobs[rows]
                if not len(rows):
                    continue
                if (np.any(self.digits[codes[rows], js] != 0)
                        or np.any(loads[server, rows] > loads[1-server, rows])):
                    raise AssertionError("Oracle state disagrees with simulator dispatch mask")
                next_codes = codes[rows] + (server+1)*self.powers[js]
                q[rows, a] = (loads[server, rows] + self.times[rows, js, server]
                              + self.values[next_codes, rows]) / self.n
        return q

    def successor(self, codes, action):
        candidates, legal = self.mdp._candidates_and_mask()
        actions = action[:, 0].numpy()
        if not legal.numpy()[self.rows, actions].all():
            raise ValueError("Illegal oracle transition")
        jobs = candidates.numpy()[self.rows, actions//2]
        return codes + (actions % 2 + 1)*self.powers[jobs]


def oracle_rollout(mdp, policy, oracle, *, collect=False, stochastic_seed=None):
    """Exact action regret on visited states; policy=None executes the oracle.

    Set-valued labels credit all equally optimal legal actions. The sum of
    Bellman regrets must equal each path's realised gap to the true optimum.
    """
    states, labels = [], []
    codes = np.zeros(mdp.nr_paths, dtype=np.int64)
    regrets = np.empty((mdp.nr_paths, mdp.nr_steps))
    optimal = np.empty_like(regrets, dtype=bool)
    legal_counts = np.empty_like(regrets, dtype=int)
    optimal_counts = np.empty_like(regrets, dtype=int)
    classes = np.full_like(regrets, -1, dtype=int)
    entropy_sum = choice_count = 0.
    with torch.random.fork_rng(devices=[]), deterministic_mode(policy, stochastic_seed is None), torch.no_grad():
        if stochastic_seed is not None:
            torch.manual_seed(stochastic_seed)
        state = mdp.reset()
        for step in range(mdp.nr_steps):
            q = oracle.action_values(codes)
            best = q.min(1)
            accepted = np.isfinite(q) & (q <= best[:, None] + oracle.tolerance)
            if collect:
                states.append(state.clone())
                labels.append(torch.from_numpy(accepted.copy()))
            action = (torch.from_numpy(q.argmin(1)[:, None]) if policy is None else policy.act(state))
            chosen = action[:, 0].numpy()
            regrets[:, step] = q[oracle.rows, chosen] - best
            optimal[:, step] = accepted[oracle.rows, chosen]
            legal_counts[:, step] = np.isfinite(q).sum(1)
            optimal_counts[:, step] = accepted.sum(1)
            choice = legal_counts[:, step] > 1
            choice_count += choice.sum()
            if hasattr(policy, "get_dist"):
                h = policy.get_dist(state).entropy().numpy()
                entropy_sum += (h[choice]/np.log(legal_counts[choice, step])).sum()
            if mdp.job_features.shape[-1]:
                candidates, _ = mdp._candidates_and_mask()
                jobs = candidates.numpy()[oracle.rows, chosen//2]
                classes[:, step] = mdp.job_features[oracle.rows, jobs].argmax(-1).numpy()
            codes = oracle.successor(codes, action)
            state, _ = mdp.step(action)
        costs = mdp.mean_sojourn().numpy().copy()
    np.testing.assert_allclose(regrets.sum(1), costs-oracle.costs, rtol=1e-9, atol=1e-9)
    result = dict(costs=costs, regrets=regrets, optimal_actions=optimal,
                  legal_counts=legal_counts, optimal_counts=optimal_counts, chosen_classes=classes,
                  entropy=entropy_sum/choice_count if choice_count else float("nan"))
    return result, (torch.cat(states), torch.cat(labels)) if collect else None


def optimal_set_loss(log_probabilities, optimal_actions):
    """Negative log probability of the entire optimal-action set, not one tie-break."""
    if not optimal_actions.any(dim=-1).all():
        raise ValueError("Every target must contain a legal optimal action")
    return -torch.logsumexp(log_probabilities.masked_fill(~optimal_actions, -torch.inf), dim=-1)


def oracle_imitation_metrics(policy, data, batch_size):
    states, targets = data
    total_loss = choice_loss = correct = choice_correct = choices = 0.
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            x, y = states[start:start+batch_size], targets[start:start+batch_size]
            choice = (x[:, -y.shape[1]:] > .5).sum(1) > 1
            if hasattr(policy, "get_dist"):
                dist = policy.get_dist(x)
                losses = optimal_set_loss(dist.base_dist.logits.squeeze(1), y)
                action = dist.mode
            else:
                action = policy.act(x)
                losses = torch.full((len(x),), float("nan"))
            accepted = y.gather(1, action).flatten()
            total_loss += losses.sum().item()
            choice_loss += losses[choice].sum().item()
            correct += accepted.sum().item()
            choice_correct += accepted[choice].sum().item()
            choices += choice.sum().item()
    return dict(loss=total_loss/len(states), choice_loss=choice_loss/choices if choices else float("nan"),
                accuracy=correct/len(states), choice_accuracy=choice_correct/choices if choices else float("nan"))


def train_oracle_epoch(policy, optimiser, data, batch_size, rng):
    """Same BC optimiser/batching as greedy imitation, with set-valued targets."""
    states, targets = data
    indices = torch.where((states[:, -targets.shape[1]:] > .5).sum(1) > 1)[0]
    indices = indices[torch.randperm(len(indices), generator=rng)]
    norms = []
    for start in tqdm(range(0, len(indices), batch_size), desc="Oracle BC minibatches", leave=False):
        batch = indices[start:start+batch_size]
        logp = policy.get_dist(states[batch]).base_dist.logits.squeeze(1)
        loss = optimal_set_loss(logp, targets[batch]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite oracle imitation loss")
        optimiser.zero_grad()
        loss.backward()
        norm = torch.stack([p.grad.detach().double().norm() for p in policy.parameters()
                            if p.grad is not None]).norm().item()
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite oracle imitation gradient")
        norms.append(norm)
        optimiser.step()
    parameter_norm(policy)
    return dict(actor_updates=len(norms), training_examples=len(indices),
                policy_grad_norm=float(np.mean(norms)) if norms else float("nan"),
                policy_grad_max=max(norms, default=float("nan")))


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
def deterministic_mode(policy, enabled=True):
    """Restore the policy's previous action-selection mode, even on failure."""
    if not hasattr(policy, "deterministic"):
        yield
        return
    old_mode = getattr(policy, "_deterministic_mode", False)
    policy.deterministic(enabled)
    try:
        yield
    finally:
        policy.deterministic(old_mode)


def evaluate(mdp: SchedulingMDP, policy, *, entropy: bool = False, agreement: bool = False,
             stochastic_seed=None):
    """Policy costs; optional seeded stochastic rollout preserves training RNG."""
    entropy_sum = 0.0
    choice_count = 0
    total_decisions = mdp.nr_paths * mdp.nr_steps
    agree_count = choice_agree_count = agreement_choices = 0
    greedy_policy = GreedyPolicy(mdp)
    with torch.random.fork_rng(devices=[]), deterministic_mode(policy, stochastic_seed is None), torch.no_grad():
        if stochastic_seed is not None:
            torch.manual_seed(stochastic_seed)
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
            if agreement:
                matches = (action == greedy_policy.act(state)).flatten()
                choices = (state[:, -mdp.n_actions:] > 0.5).sum(dim=1) > 1
                agree_count += int(matches.sum())
                choice_agree_count += int(matches[choices].sum())
                agreement_choices += int(choices.sum())
            state, reward = mdp.step(action)
            rewards += reward.double()
        costs = mdp.mean_sojourn()
        torch.testing.assert_close(-rewards, costs, rtol=1e-5, atol=1e-5)
    result = dict(costs=costs.cpu().numpy().copy())
    if entropy:
        result["entropy"] = entropy_sum / choice_count if choice_count else float("nan")
        result["choice_fraction"] = choice_count / total_decisions
    if agreement:
        result["greedy_agreement"] = agree_count / total_decisions
        result["greedy_choice_agreement"] = (choice_agree_count / agreement_choices
                                              if agreement_choices else float("nan"))
    return result


def make_algorithm(mdp, name, seed, actor=None, *, batch_size=None, updates=None):
    """Identical initial actor/critic seeds and fresh optimiser states per arm."""
    torch.manual_seed(seed)
    overrides = {}
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if updates is not None:
        overrides["update_per_rollout"] = updates
    algorithm = get_algorithm(
        name, mdp, policy_type="categorical", mask_from_state=True,
        gamma=1.0, gae_lambda=1.0, **overrides,
    )
    if actor is not None:
        algorithm.policy.load_state_dict(actor.state_dict())
    return algorithm


def warmup_critic(algorithm, steps, batch_size, seed, directory):
    """Fit only V to undiscounted MC returns from one fixed-policy training rollout.

    The warm-up consumes no PPO RNG, updates no actor weights, and leaves both
    PPO Adam states empty. Its training data never include validation/test paths.
    """
    started, cpu_started = perf_counter(), process_time()
    mdp, policy, critic = algorithm.mdp, algorithm.policy, algorithm.value_function
    actor_before = deepcopy(policy.state_dict())
    flags = [p.requires_grad for p in policy.parameters()]
    rows = []
    directory.mkdir(parents=True, exist_ok=True)
    try:
        policy.requires_grad_(False)
        with torch.random.fork_rng(devices=[]), deterministic_mode(policy, False):
            torch.manual_seed(seed)
            with torch.no_grad():
                states, rewards = [], []
                state = mdp.reset()
                for _ in range(mdp.nr_steps):
                    states.append(state.clone())
                    state, reward = mdp.step(policy.act(state))
                    rewards.append(reward)
                reward_matrix = torch.stack(rewards, dim=1)
                returns = reward_matrix.flip(1).cumsum(1).flip(1)
                torch.testing.assert_close(-returns[:, 0].double(), mdp.mean_sojourn(), rtol=1e-5, atol=1e-5)
                x = torch.stack(states, dim=1).reshape(-1, mdp.state_dim)
                y = returns.reshape(-1)
            # Use the usual critic LR and clipping, then discard its Adam moments.
            with gradient_monitor(algorithm) as (record, clipped):
                for step in tqdm(range(steps + 1), desc="Critic warm-up"):
                    if step:
                        indices = torch.randint(len(x), (batch_size,))
                        loss = torch.nn.functional.mse_loss(critic.value(x[indices]).flatten(), y[indices])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite critic warm-up loss")
                        algorithm.value_function_optim.zero_grad()
                        loss.backward()
                        if algorithm.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), algorithm.max_grad_norm)
                        algorithm.value_function_optim.step()
                    if step % 100 == 0 or step == steps:
                        with torch.no_grad():
                            mse_sum = sum((critic.value(a).flatten()-b).square().sum().item()
                                          for a, b in zip(x.split(4096), y.split(4096)))
                        rows.append(dict(step=step, training_mse=mse_sum/len(x), seconds=perf_counter()-started))
                        print(f"Warm-up {step}/{steps}: MC training MSE {mse_sum/len(x):.5f}", flush=True)
            gradient_summary(record, clipped)
    finally:
        for parameter, flag in zip(policy.parameters(), flags):
            parameter.requires_grad_(flag)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    assert not algorithm.policy_optim.state
    algorithm.value_function_optim.state.clear()
    parameter_norm(critic)
    with (directory / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(critic_warmup_seconds=perf_counter()-started,
                critic_warmup_cpu_seconds=process_time()-cpu_started,
                critic_warmup_updates=steps, critic_warmup_examples=steps*batch_size,
                critic_warmup_rollout_decisions=mdp.nr_paths*mdp.nr_steps)


def collect_demonstrations(mdp):
    """Greedy observations and legal labels, ordered by decision then path."""
    greedy = GreedyPolicy(mdp)
    states, actions = [], []
    with torch.no_grad():
        state = mdp.reset()
        rewards = torch.zeros(mdp.nr_paths, dtype=torch.float64)
        for _ in range(mdp.nr_steps):
            action = greedy.act(state)
            if not (state[:, -mdp.n_actions:].gather(1, action) > 0.5).all():
                raise AssertionError("Illegal greedy demonstration")
            states.append(state.clone())
            actions.append(action.clone())
            state, reward = mdp.step(action)
            rewards += reward.double()
        torch.testing.assert_close(-rewards, mdp.mean_sojourn(), rtol=1e-5, atol=1e-5)
    return torch.cat(states), torch.cat(actions)


def imitation_metrics(policy, data, batch_size):
    """Offline metrics on greedy-visited states, including forced decisions."""
    states, actions = data
    loss_sum = choice_loss = correct = choice_correct = n_choices = 0
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state, action = states[start:start + batch_size], actions[start:start + batch_size]
            dist = policy.get_dist(state)
            losses = -dist.log_prob(action)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite imitation loss")
            matches = (dist.mode == action).flatten()
            choices = (state[:, -policy.n_actions:] > 0.5).sum(dim=1) > 1
            loss_sum += losses.sum().item()
            choice_loss += losses[choices].sum().item()
            correct += int(matches.sum())
            choice_correct += int(matches[choices].sum())
            n_choices += int(choices.sum())
    return dict(cross_entropy=loss_sum / len(states), accuracy=correct / len(states),
                choice_cross_entropy=choice_loss / n_choices if n_choices else float("nan"),
                choice_accuracy=choice_correct / n_choices if n_choices else float("nan"),
                n_samples=len(states), n_choice_samples=n_choices)


def train_bc_epoch(policy, optimiser, data, batch_size, rng):
    """One Adam epoch over non-forced training demonstrations only."""
    states, actions = data
    indices = torch.where((states[:, -policy.n_actions:] > 0.5).sum(dim=1) > 1)[0]
    indices = indices[torch.randperm(len(indices), generator=rng)]
    norms = []
    for start in tqdm(range(0, len(indices), batch_size), desc="BC minibatches", leave=False):
        batch = indices[start:start + batch_size]
        loss = -policy.get_logprobs(states[batch], actions[batch]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BC loss")
        optimiser.zero_grad()
        loss.backward()
        norm = torch.stack([p.grad.detach().double().norm()
                            for p in policy.parameters() if p.grad is not None]).norm().item()
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite BC gradient")
        norms.append(norm)
        optimiser.step()
    parameter_norm(policy)
    stats = gradient_summary({"policy": norms, "critic": []}, {"policy": 0, "critic": 0})
    # No gradient clipping or regularisation in BC.
    return dict(**stats, actor_updates=len(norms), training_examples=len(indices))


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


def plot_history(rows: list[dict], directory: Path):
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
        plt.savefig(directory / f"{name}.png", dpi=150)
        plt.close()

    def col(name):
        return [r[name] for r in rows]

    save_plot("cost", "Deterministic scheduling performance", "Mean sojourn time", [
        ("Training", col("train_cost"), "-"),
        ("Validation", col("test_cost"), "-"),
        ("Greedy", col("greedy_cost"), "--"),
        ("Offline bound", col("oracle_cost"), ":"),
    ])
    save_plot("entropy", "Policy entropy on states with multiple legal actions", "Normalised entropy", [
        ("Validation entropy", col("entropy"), "-"),
    ])
    save_plot("critic", "Value-function diagnostics", "Explained variance", [
        ("Validation critic EV", col("critic_ev"), "-"),
    ])
    save_plot("gradients", "Pre-clipping gradient norms", "Mean gradient L2 norm", [
        ("Policy", col("policy_grad_norm"), "-"),
        ("Critic", col("critic_grad_norm"), "-"),
    ])


def plot_comparison(rows, directory):
    """Compare validation costs; epochs are deliberately not plotted as rounds."""
    rows = [r for r in rows if r["split"] == "validation"]
    plt.figure(figsize=(8, 4.5))
    for name, label in (("ppo", "PPO scratch"), ("bc-ppo", "BC → PPO")):
        selected = [r for r in rows if r["experiment"] == name]
        if selected:
            plt.plot([r["round"] for r in selected], [r["test_cost"] for r in selected], label=label)
    bc = [r for r in rows if r["experiment"] == "bc"]
    if bc:
        plt.axhline(bc[-1]["test_cost"], color="C2", linestyle="-.", label="Final BC")
    for key, label, style, colour in (("greedy_cost", "Greedy", "--", "C3"),
                                     ("oracle_cost", "Offline bound", ":", "C4")):
        plt.axhline(rows[0][key], color=colour, linestyle=style, label=label)
    plt.xlabel("PPO training rounds (BC pretraining effort is recorded in CSV)")
    plt.ylabel("Validation mean sojourn time")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(directory / "comparison.png", dpi=150)
    plt.close()


def run_oracle_experiment(args, directory):
    """One fixed oracle-BC fit; reuse trained controls, never tune on final data."""
    started = perf_counter()
    progress = json.loads((args.comparison_suite / "progress.json").read_text())
    references = {}
    for name, arm, method, step in (
        ("greedy-bc", "bc-ppo", "bc", f"epoch-{args.bc_epochs:05d}"),
        ("scratch-ppo", "ppo-b512-u16", "ppo", f"round-{args.n_rounds:05d}"),
        ("bc-ppo", "bc-ppo", "bc-ppo", f"round-{args.n_rounds:05d}"),
    ):
        child = Path(progress[arm]["directory"])
        settings = json.loads((child / "settings.json").read_text())
        for key in ("seed", "scenario_seed", "n_paths", "n_test_paths", "n_jobs",
                    "n_rounds", "bc_epochs", "bc_lr", "bc_batch_size", "test_distribution"):
            if settings["settings"][key] != getattr(args, key):
                raise ValueError(f"Comparison {name} differs in {key}; use the established settings")
        if settings["scenario_seeds"]["final_test"] != args.final_test_seed:
            raise ValueError("Comparison uses a different final-test workload seed")
        path = child / method / "validation" / step / "weights.pt"
        references[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    protocol = dict(seed=args.seed, training_paths=args.n_paths, validation_paths=args.n_test_paths,
                    training_seed=args.scenario_seed, validation_seed=args.scenario_seed+1,
                    final_test_seed=args.final_test_seed, final_test_paths=args.n_test_paths,
                    oracle="Exact ternary-subset dynamic program; legal non-idling dispatches only",
                    tie_tolerance=ExactScheduler.tolerance,
                    loss="Negative log probability mass on the set of all optimal actions; choice states only",
                    selection=f"Fixed final epoch {args.bc_epochs}; no checkpoint or hyperparameter selection",
                    comparisons=references, test_reuse="Same independent test scenarios as previous study; "
                    "previous aggregate results were already observed, so this is not a fresh untouched confirmation")
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def write_rows(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def prepare_split(split, count, seed):
        mdp = make_mdp(2, count, 8, seed, "loguniform" if split == "training" else args.test_distribution)
        oracle = ExactScheduler(mdp, f"Exact oracle: {split}")
        teacher, data = oracle_rollout(mdp, None, oracle, collect=True)
        greedy = evaluate(mdp, GreedyPolicy(mdp))["costs"]
        lower = optimal_costs(mdp)
        if np.any(lower > oracle.costs+1e-9) or np.any(oracle.costs > greedy+1e-9):
            raise AssertionError("Expected relaxed bound <= exact oracle <= greedy pathwise")
        target = directory / split
        target.mkdir()
        np.savez_compressed(target / "oracle.npz", path_index=np.arange(count), exact_oracle=oracle.costs,
                            greedy=greedy, relaxed_bound=lower, service_times=mdp.service_times.numpy(),
                            **teacher)
        np.savez_compressed(target / "demonstrations.npz", states=data[0].numpy(), optimal_actions=data[1].numpy())
        print(f"{split}: exact {oracle.costs.mean():.6f}; relaxed {lower.mean():.6f}; "
              f"greedy {greedy.mean():.6f}; equivalent-action labels saved", flush=True)
        return mdp, oracle, data, greedy

    train, train_oracle, train_data, train_greedy = prepare_split("training", args.n_paths, args.scenario_seed)
    validation, val_oracle, val_data, val_greedy = prepare_split("validation", args.n_test_paths, args.scenario_seed+1)
    policy = make_algorithm(train, "ppo", args.seed).policy
    initial_path = Path(progress["bc-ppo"]["directory"]) / "bc" / "validation" / "epoch-00000" / "weights.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)["policy"]
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.bc_lr)
    rng = torch.Generator().manual_seed(args.seed)
    history, comparisons = [], []
    training_seconds = 0.
    actor_updates = training_examples = 0
    stats = dict(policy_grad_norm=float("nan"), policy_grad_max=float("nan"))
    for epoch in range(args.bc_epochs+1):
        if epoch:
            tick = perf_counter()
            stats = train_oracle_epoch(policy, optimiser, train_data, args.bc_batch_size, rng)
            training_seconds += perf_counter()-tick
            actor_updates += stats.pop("actor_updates")
            training_examples += stats.pop("training_examples")
        print(f"Evaluating oracle imitation epoch {epoch}/{args.bc_epochs}...", flush=True)
        train_result, _ = oracle_rollout(train, policy, train_oracle)
        result, _ = oracle_rollout(validation, policy, val_oracle)
        stochastic, _ = oracle_rollout(validation, policy, val_oracle, stochastic_seed=args.seed+200003)
        metrics = {f"{part}_{key}": value for part, data in (("train", train_data), ("validation", val_data))
                   for key, value in oracle_imitation_metrics(policy, data, args.bc_batch_size).items()}
        choice = result["legal_counts"] > 1
        gap = result["costs"] - val_oracle.costs
        row = dict(epoch=epoch, train_cost=float(train_result["costs"].mean()),
                   test_cost=float(result["costs"].mean()), stochastic_cost=float(stochastic["costs"].mean()),
                   exact_oracle_cost=float(val_oracle.costs.mean()), greedy_cost=float(val_greedy.mean()),
                   optimality_gap=float(gap.mean()), paired_se=paired_se(gap),
                   closed_loop_choice_accuracy=float(result["optimal_actions"][choice].mean()),
                   train_closed_loop_choice_accuracy=float(train_result["optimal_actions"][train_result["legal_counts"]>1].mean()),
                   optimal_trajectory_fraction=float(np.mean(gap <= ExactScheduler.tolerance)),
                   entropy=float(result["entropy"]), train_seconds=training_seconds,
                   actor_updates=actor_updates, training_examples=training_examples,
                   wall_seconds=perf_counter()-started, **metrics, **stats)
        history.append(row)
        target = directory / "oracle-bc" / "validation" / f"epoch-{epoch:05d}"
        target.mkdir(parents=True)
        torch.save(dict(policy=policy.state_dict(), epoch=epoch, seed=args.seed, scenario_seed=args.scenario_seed,
                        stage=2, n_jobs=8, experiment="oracle-bc"), target / "weights.pt")
        np.savez_compressed(target / "costs.npz", path_index=np.arange(validation.nr_paths),
                            deterministic=result["costs"], stochastic=stochastic["costs"],
                            exact_oracle=val_oracle.costs, greedy=val_greedy,
                            regrets=result["regrets"], optimal_actions=result["optimal_actions"],
                            legal_counts=result["legal_counts"], optimal_counts=result["optimal_counts"])
        write_rows(directory / "diagnostics.csv", history)
        print(f"Oracle BC epoch {epoch}: train {row['train_cost']:.6f}, validation {row['test_cost']:.6f}, "
              f"gap {gap.mean():.6f}, offline/closed-loop optimal choices "
              f"{row['validation_choice_accuracy']:.1%}/{row['closed_loop_choice_accuracy']:.1%}", flush=True)

    # Fix the final epoch and weight provenance before generating final-test data.
    final_path = directory / "oracle-bc" / "validation" / f"epoch-{args.bc_epochs:05d}" / "weights.pt"
    references["oracle-bc"] = dict(path=str(final_path.resolve()),
                                  sha256=hashlib.sha256(final_path.read_bytes()).hexdigest())
    (directory / "selection.json").write_text(json.dumps(references, indent=2), encoding="utf-8")

    def compare(split, mdp, oracle, data, greedy):
        for name in ("exact-oracle", "greedy", *references):
            weights = None
            if name in references:
                weights = torch.load(references[name]["path"], map_location="cpu", weights_only=True)
                actor = make_algorithm(mdp, "ppo", args.seed).policy
                actor.load_state_dict(weights["policy"])
                offline = oracle_imitation_metrics(actor, data, args.bc_batch_size)
            elif name == "greedy":
                actor = GreedyPolicy(mdp)
                offline = oracle_imitation_metrics(actor, data, args.bc_batch_size)
            else:
                actor = None
                offline = dict(choice_accuracy=1.)
            for mode in ("deterministic", "stochastic"):
                print(f"Comparing {split}: {name}, {mode}...", flush=True)
                action_seed = args.seed + (300003 if split == "final_test" else 200003)
                r, _ = oracle_rollout(mdp, actor, oracle, stochastic_seed=action_seed if mode == "stochastic" else None)
                gap, delta = r["costs"]-oracle.costs, r["costs"]-greedy
                choice = r["legal_counts"] > 1
                comparisons.append(dict(split=split, policy=name, mode=mode, cost=float(r["costs"].mean()),
                    oracle_cost=float(oracle.costs.mean()), gap=float(gap.mean()), gap_se=paired_se(gap),
                    delta_greedy=float(delta.mean()), delta_greedy_se=paired_se(delta),
                    offline_choice_accuracy=offline["choice_accuracy"],
                    closed_loop_choice_accuracy=float(r["optimal_actions"][choice].mean()),
                    first_choice_accuracy=float(r["optimal_actions"][:, 0].mean()),
                    equivalent_choice_fraction=float((r["optimal_counts"][choice]>1).mean()),
                    optimal_trajectory_fraction=float(np.mean(gap <= ExactScheduler.tolerance)), entropy=float(r["entropy"])))
                target = directory / name / split / mode
                target.mkdir(parents=True)
                np.savez_compressed(target / "costs.npz", path_index=np.arange(mdp.nr_paths),
                                    exact_oracle=oracle.costs, greedy=greedy, **r)
            if weights is not None:
                torch.save(weights, directory / name / split / "weights.pt")
            write_rows(directory / "comparison.csv", comparisons)

    compare("validation", validation, val_oracle, val_data, val_greedy)
    del train_oracle, val_oracle, train_data, val_data
    test, test_oracle, test_data, test_greedy = prepare_split("final_test", args.n_test_paths, args.final_test_seed)
    compare("final_test", test, test_oracle, test_data, test_greedy)
    if not args.no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [r["epoch"] for r in history]
        for key, label in (("train_cost", "Training"), ("test_cost", "Validation"), ("stochastic_cost", "Stochastic validation")):
            axes[0].plot(epochs, [r[key] for r in history], label=label)
        axes[0].axhline(history[0]["exact_oracle_cost"], color="black", ls="--", label="Exact optimum")
        for key, label in (("train_choice_accuracy", "Training teacher states"),
                           ("validation_choice_accuracy", "Validation teacher states"),
                           ("closed_loop_choice_accuracy", "Validation visited states")):
            axes[1].plot(epochs, [r[key] for r in history], label=label)
        for ax in axes:
            ax.set_xlabel("Oracle BC epoch")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Mean sojourn cost")
        axes[1].set_ylabel("Optimal-action accuracy (equivalent actions credited)")
        fig.tight_layout()
        fig.savefig(directory / "learning_curves.png", dpi=180)
        plt.close(fig)
    (directory / "completion.json").write_text(json.dumps(dict(status="completed", epochs=args.bc_epochs,
        actor_updates=actor_updates, training_examples=training_examples, training_seconds=training_seconds,
        wall_seconds=perf_counter()-started), indent=2), encoding="utf-8")
    print(f"Oracle study complete: {directory}", flush=True)


def run_optimisation_suite(args, directory):
    """Run the specified seven arms; select on validation before opening test data."""
    grid = [(128, 2), (512, 2), (128, 8), (512, 8), (512, 16)]
    plans = [dict(name=f"ppo-b{batch}-u{updates}", experiment="ppo", batch=batch, updates=updates, warmup=0)
             for batch, updates in grid]
    plans += [dict(name="bc-ppo", experiment="bc-ppo", batch=128, updates=2, warmup=0),
              dict(name="bc-ppo-warmup", experiment="bc-ppo", batch=128, updates=2,
                   warmup=args.critic_warmup_steps or 1000)]
    protocol = dict(arms=plans, seed=args.seed, scenario_seed=args.scenario_seed,
                    selection="Lowest deterministic validation cost at final round among five scratch arms; "
                              "ties: fewer updates then smaller batch. Test winner, baseline, both BC-PPO arms and BC.",
                    final_test_seed=args.final_test_seed, final_test_paths=args.n_test_paths,
                    stochastic_evaluation="One seeded stochastic trajectory per scenario; paired by path index.",
                    critic_warmup="One stochastic training rollout, undiscounted MC returns; actor frozen; "
                                  "configured critic LR and clipping; reset critic Adam state before PPO.",
                    workers=args.suite_workers, threads_per_worker=1)
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    active, completed, pending = [], {}, list(plans)
    progress = {}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}
    last_status = 0.
    try:
        while pending or active:
            while pending and len(active) < args.suite_workers:
                plan = pending.pop(0)
                run_name = args.run_name + "-" + plan["name"]
                child_dir = directory.with_name(run_name)
                child_dir.mkdir(exist_ok=False)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "2",
                           "--experiment", plan["experiment"], "--n-paths", str(args.n_paths),
                           "--n-test-paths", str(args.n_test_paths), "--n-jobs", str(args.n_jobs),
                           "--n-rounds", str(args.n_rounds), "--eval-every", str(args.eval_every),
                           "--seed", str(args.seed), "--scenario-seed", str(args.scenario_seed),
                           "--final-test-seed", str(args.final_test_seed),
                           "--bc-epochs", str(args.bc_epochs), "--bc-lr", str(args.bc_lr),
                           "--bc-batch-size", str(args.bc_batch_size),
                           "--ppo-batch-size", str(plan["batch"]), "--ppo-updates", str(plan["updates"]),
                           "--critic-warmup-steps", str(plan["warmup"]),
                           "--critic-warmup-batch-size", str(args.critic_warmup_batch_size),
                           "--test-distribution", args.test_distribution,
                           "--output-date", directory.parent.name,
                           "--output-prefix", str(directory.parent.parent / run_name)]
                if args.no_plots:
                    command.append("--no-plots")
                stdout = (child_dir / "stdout.log").open("w", encoding="utf-8")
                stderr = (child_dir / "stderr.log").open("w", encoding="utf-8")
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active.append((plan, child_dir, process, stdout, stderr))
                progress[plan["name"]] = dict(status="running", pid=process.pid, directory=str(child_dir), command=command)
                print(f"Started {plan['name']} (PID {process.pid})", flush=True)
            for job in active[:]:
                plan, child_dir, process, stdout, stderr = job
                code = process.poll()
                if code is not None:
                    stdout.close()
                    stderr.close()
                    progress[plan["name"]].update(status="completed" if code == 0 else "failed", exit_code=code)
                    active.remove(job)
                    if code != 0:
                        (directory / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                        raise RuntimeError(f"{plan['name']} failed with exit {code}; see {child_dir / 'stderr.log'}")
                    completed[plan["name"]] = child_dir
                    print(f"Completed {plan['name']}", flush=True)
            if perf_counter() - last_status > 30:
                for plan, child_dir, *_ in active:
                    path = child_dir / "diagnostics.csv"
                    if path.exists():
                        with path.open(encoding="utf-8") as handle:
                            current = list(csv.DictReader(handle))
                        if current and current[-1].get("round"):
                            r = current[-1]
                            print(f"Progress {plan['name']}: {r['experiment']} epoch {r['epoch']}, "
                                  f"round {r['round']}, validation {r['test_cost']}", flush=True)
                last_status = perf_counter()
            (directory / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            if pending or active:
                sleep(3)
    finally:
        for _, _, process, _, _ in active:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        for plan, _, process, stdout, stderr in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                stdout.close()
                stderr.close()
            progress[plan["name"]].update(status="cancelled", exit_code=process.returncode)
        (directory / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    final_rows = {}
    for plan in plans:
        with (completed[plan["name"]] / "diagnostics.csv").open(encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["experiment"] == plan["experiment"]]
        assert int(rows[-1]["round"]) == args.n_rounds
        final_rows[plan["name"]] = rows[-1]
    winner = min(plans[:5], key=lambda p: (float(final_rows[p["name"]]["test_cost"]), p["updates"], p["batch"]))["name"]
    selected_names = list(dict.fromkeys(["ppo-b128-u2", winner, "bc-ppo", "bc-ppo-warmup"]))
    selected = {name: completed[name] / ("ppo" if name.startswith("ppo-") else "bc-ppo") /
                "validation" / f"round-{args.n_rounds:05d}" / "weights.pt" for name in selected_names}
    selected["bc"] = completed["bc-ppo"] / "bc" / "validation" / f"epoch-{args.bc_epochs:05d}" / "weights.pt"
    selection = dict(winner=winner, rule=protocol["selection"], weights={k: str(v) for k, v in selected.items()},
                     validation_final_costs={k: float(v["test_cost"]) for k, v in final_rows.items()})
    (directory / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selection locked: {winner}. Opening untouched final-test scenarios once.", flush=True)
    test = make_mdp(2, args.n_test_paths, args.n_jobs, args.final_test_seed, args.test_distribution)
    greedy = evaluate(test, GreedyPolicy(test))["costs"]
    oracle = optimal_costs(test)
    result_rows = []
    for name, path in selected.items():
        weights = torch.load(path, map_location="cpu", weights_only=True)
        algorithm = make_algorithm(test, "ppo", args.seed)
        algorithm.policy.load_state_dict(weights["policy"])
        destination = directory / "final_test" / name
        destination.mkdir(parents=True, exist_ok=False)
        costs = {}
        for mode in ("deterministic", "stochastic"):
            result = evaluate(test, algorithm.policy, entropy=True, agreement=True,
                              stochastic_seed=args.seed + 300003 if mode == "stochastic" else None)
            costs[mode] = result["costs"]
            difference = result["costs"] - greedy
            result_rows.append(dict(configuration=name, mode=mode, round=weights["round"], epoch=weights["epoch"],
                                    mean_sojourn=float(result["costs"].mean()), greedy_cost=float(greedy.mean()),
                                    oracle_cost=float(oracle.mean()), delta_greedy=float(difference.mean()),
                                    paired_se=paired_se(difference), entropy=result["entropy"],
                                    greedy_choice_agreement=result["greedy_choice_agreement"]))
        np.savez_compressed(destination / "costs.npz", path_index=np.arange(test.nr_paths),
                            greedy=greedy, oracle=oracle, **costs)
        torch.save(weights, destination / "weights.pt")
        print(f"Final test {name}: deterministic {costs['deterministic'].mean():.5f}, "
              f"stochastic {costs['stochastic'].mean():.5f}", flush=True)
    with (directory / "final_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"All seven configurations and selected final-test evaluations completed: {directory}", flush=True)


def main() -> None:
    args = parse_arguments()
    run_started = perf_counter()
    output_dir = prepare_output(args)
    if args.experiment == "oracle":
        run_oracle_experiment(args, output_dir)
        return
    if args.experiment == "optimisation":
        run_optimisation_suite(args, output_dir)
        return
    print("Generating fixed training/validation scenarios and baselines...", flush=True)
    setup_start = perf_counter()
    train = make_mdp(args.stage, args.n_paths, args.n_jobs, args.scenario_seed)
    test = make_mdp(args.stage, args.n_test_paths, args.n_jobs, args.scenario_seed + 1,
                    args.test_distribution)
    greedy = evaluate(test, GreedyPolicy(test))["costs"]
    oracle = optimal_costs(test)
    if np.any(greedy + 1e-5 < oracle):
        raise AssertionError("Greedy beat the offline optimum")
    if args.stage == 1:
        np.testing.assert_allclose(greedy, oracle, atol=1e-5, rtol=1e-6)

    print(f"Stage {args.stage}: {args.n_jobs} jobs, {train.n_servers} server(s)")
    print(f"Fixed training/validation scenarios: {args.n_paths:,} / {args.n_test_paths:,}")
    print(f"Policy seed: {args.seed}; scenario seeds: {args.scenario_seed}, "
          f"{args.scenario_seed + 1}, {args.final_test_seed} (reserved final test)")
    print(f"Validation/final-test distribution: {args.test_distribution}")
    print(f"Greedy mean: {greedy.mean():.4f}; offline bound: {oracle.mean():.4f}")
    print(f"Scenario/baseline setup: {perf_counter() - setup_start:.2f}s", flush=True)
    print("Method Epoch Round  Train    Validation  DeltaGreedy +/- SE   Hnorm  CriticEV  Train(s)")

    rows = []
    csv_path = output_dir / "diagnostics.csv"

    split = "validation"
    demo_seconds = 0.0
    metric_names = ("cross_entropy", "accuracy", "choice_cross_entropy", "choice_accuracy",
                    "n_samples", "n_choice_samples")

    def checkpoint(name, policy, algorithm, round_number, epoch, stats, effort, datasets=None):
        started = perf_counter()
        print(f"Evaluating {split}: {name}, epoch {epoch}, round {round_number}...", flush=True)
        imitation = {}
        if datasets:
            for part, data in zip(("train", "validation"), datasets):
                imitation.update({f"bc_{part}_{key}": value for key, value in
                                  imitation_metrics(policy, data, args.bc_batch_size).items()})
        train_result = evaluate(train, policy)
        test_result = evaluate(test, policy, entropy=True, agreement=True)
        stochastic = evaluate(test, policy, stochastic_seed=args.seed + (300003 if split == "final_test" else 200003))
        cost = test_result["costs"]
        diff = cost - greedy
        row = dict(
            seed=args.seed, scenario_seed=args.scenario_seed, experiment=name, split=split,
            epoch=epoch,
            round=round_number,
            train_cost=float(train_result["costs"].mean()),
            test_cost=float(cost.mean()),
            greedy_cost=float(greedy.mean()),
            oracle_cost=float(oracle.mean()),
            delta_greedy=float(diff.mean()),
            paired_se=paired_se(diff),
            stochastic_cost=float(stochastic["costs"].mean()),
            stochastic_delta_greedy=float((stochastic["costs"] - greedy).mean()),
            stochastic_paired_se=paired_se(stochastic["costs"] - greedy),
            stochastic_minus_deterministic=float((stochastic["costs"] - cost).mean()),
            stochastic_minus_deterministic_se=paired_se(stochastic["costs"] - cost),
            delta_oracle=float((cost - oracle).mean()),
            excess_pct=100.0 * (cost.mean() / oracle.mean() - 1.0),
            entropy=float(test_result["entropy"]),
            choice_fraction=float(test_result["choice_fraction"]),
            greedy_agreement=test_result["greedy_agreement"],
            greedy_choice_agreement=test_result["greedy_choice_agreement"],
            critic_ev=critic_explained_variance(test, policy,
                                                algorithm.value_function,
                                                seed=args.seed + 100003) if algorithm else float("nan"),
            policy_param_norm=parameter_norm(policy),
            critic_param_norm=parameter_norm(algorithm.value_function) if algorithm else float("nan"),
            **({f"{part}_{key}": float("nan") for part in ("bc_train", "bc_validation")
                for key in metric_names} | (imitation or {})),
            **effort,
            train_seconds=effort["bc_seconds"] + effort["ppo_seconds"] + effort["critic_warmup_seconds"],
            batch_size=algorithm.batch_size if algorithm else args.bc_batch_size,
            updates_per_rollout=algorithm.update_per_rollout if algorithm else 0,
            wall_seconds=perf_counter() - run_started,
            demonstration_seconds=demo_seconds if name in ("bc", "bc-ppo") else 0.,
            diagnostic_seconds=perf_counter() - started,
            **stats,
        )
        essential = (row["train_cost"], row["test_cost"],
                     row["policy_param_norm"])
        if algorithm:
            essential += (row["critic_param_norm"],)
        if not np.isfinite(essential).all():
            raise FloatingPointError(f"Non-finite parameters or costs at round {round_number}")
        # Entropy and EV can be undefined when there is no action choice or
        # essentially no variation in realised returns, respectively.
        if row["choice_fraction"] > 0 and not np.isfinite(row["entropy"]):
            raise FloatingPointError(f"Non-finite policy entropy at round {round_number}")
        step_name = f"epoch-{epoch:05d}" if name == "bc" else f"round-{round_number:05d}"
        artifact_dir = output_dir / name / split / step_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact_dir / "costs.npz", path_index=np.arange(test.nr_paths),
                            deterministic=cost, stochastic=stochastic["costs"], greedy=greedy, oracle=oracle)
        torch.save(dict(policy=policy.state_dict(), critic=algorithm.value_function.state_dict() if algorithm else None,
                        seed=args.seed, scenario_seed=args.scenario_seed, round=round_number, epoch=epoch,
                        experiment=name, stage=args.stage, n_jobs=args.n_jobs), artifact_dir / "weights.pt")
        rows.append(row)
        # Save at every checkpoint, so an interrupted run retains its history.
        with csv_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name:6s} {epoch:5d} {round_number:5d} {row['train_cost']:8.4f} {row['test_cost']:8.4f} "
              f"{row['delta_greedy']:+9.4f} +/-{row['paired_se']:.4f} "
              f"{row['entropy']:7.3f} {row['critic_ev']:9.3f} {row['train_seconds']:9.2f}", flush=True)
        if imitation:
            print(f"  Offline BC train/validation CE: {row['bc_train_cross_entropy']:.4f} / "
                  f"{row['bc_validation_cross_entropy']:.4f}; accuracy: "
                  f"{row['bc_train_accuracy']:.1%} / {row['bc_validation_accuracy']:.1%}; "
                  f"choice accuracy: {row['bc_train_choice_accuracy']:.1%} / "
                  f"{row['bc_validation_choice_accuracy']:.1%}; closed-loop choice agreement: "
                  f"{row['greedy_choice_agreement']:.1%}", flush=True)
        return test_result

    empty = gradient_summary({"policy": [], "critic": []}, {"policy": 0, "critic": 0})
    no_effort = dict(bc_seconds=0., ppo_seconds=0., ppo_cpu_seconds=0., actor_updates=0, training_examples=0,
                     ppo_actor_updates=0, ppo_critic_updates=0, ppo_sampled_transitions=0, ppo_rollout_decisions=0,
                     critic_warmup_seconds=0., critic_warmup_cpu_seconds=0., critic_warmup_updates=0,
                     critic_warmup_examples=0, critic_warmup_rollout_decisions=0,
                     critic_ev_before_warmup=float("nan"))
    finals = []
    bc_policy = None
    if args.experiment in ("bc", "bc-ppo", "all"):
        print("Collecting greedy training and validation demonstrations...", flush=True)
        started = perf_counter()
        training_data = collect_demonstrations(train)
        validation_data = collect_demonstrations(test)
        np.testing.assert_array_equal(test.mean_sojourn().numpy(), greedy)
        demo_seconds = perf_counter() - started
        print(f"Collected {len(training_data[0]):,} / {len(validation_data[0]):,} pairs "
              f"in {demo_seconds:.2f}s", flush=True)
        bc_policy = deepcopy(make_algorithm(train, "ppo", args.seed).policy)
        optimiser = torch.optim.Adam(bc_policy.parameters(), lr=args.bc_lr)
        rng = torch.Generator().manual_seed(args.seed)
        bc_effort = no_effort.copy()
        stats = empty
        for epoch in range(args.bc_epochs + 1):
            if epoch:
                started = perf_counter()
                stats = train_bc_epoch(bc_policy, optimiser, training_data, args.bc_batch_size, rng)
                bc_effort["bc_seconds"] += perf_counter() - started
                for key in ("actor_updates", "training_examples"):
                    bc_effort[key] += stats.pop(key)
            bc_result = checkpoint("bc", bc_policy, None, 0, epoch, stats, bc_effort,
                                   (training_data, validation_data))
        finals.append(("bc", bc_policy, None, 0, args.bc_epochs, stats, bc_effort.copy()))
        print("Compare offline choice accuracy with closed-loop choice agreement and paired cost "
              "to diagnose distribution shift; accuracy alone is not scheduling performance.", flush=True)

    experiments = (["ppo", "bc-ppo"] if args.experiment == "all" else
                   [] if args.experiment == "bc" else [args.experiment or args.algorithm])
    for name in experiments:
        warm = name == "bc-ppo"
        algorithm = make_algorithm(train, args.algorithm, args.seed, bc_policy if warm else None,
                                   batch_size=args.ppo_batch_size, updates=args.ppo_updates)
        effort = bc_effort.copy() if warm else no_effort.copy()
        epoch = args.bc_epochs if warm else 0
        if warm and args.critic_warmup_steps:
            effort["critic_ev_before_warmup"] = critic_explained_variance(
                test, algorithm.policy, algorithm.value_function, args.seed + 100003)
            effort.update(warmup_critic(algorithm, args.critic_warmup_steps, args.critic_warmup_batch_size,
                                       args.seed + 100019, output_dir / "warmup"))
        initial = checkpoint(name, algorithm.policy, algorithm, 0, epoch, empty, effort)
        if warm:
            # Check both actions and per-path costs, not only the mean.
            with deterministic_mode(bc_policy), deterministic_mode(algorithm.policy), torch.no_grad():
                for batch in validation_data[0].split(args.bc_batch_size):
                    torch.testing.assert_close(bc_policy.act(batch), algorithm.policy.act(batch), rtol=0, atol=0)
            np.testing.assert_array_equal(initial["costs"], bc_result["costs"])
            print("BC -> PPO round 0: deterministic actions and paired costs match exactly.", flush=True)
        stats = empty
        for start in range(0, args.n_rounds, args.eval_every):
            n = min(args.eval_every, args.n_rounds - start)
            started = perf_counter()
            cpu_started = process_time()
            with gradient_monitor(algorithm) as (records, clipped):
                algorithm.train(n)
            effort["ppo_seconds"] += perf_counter() - started
            effort["ppo_cpu_seconds"] += process_time() - cpu_started
            effort["actor_updates"] += n * algorithm.update_per_rollout
            effort["training_examples"] += n * algorithm.update_per_rollout * algorithm.batch_size
            effort["ppo_actor_updates"] += n * algorithm.update_per_rollout
            effort["ppo_critic_updates"] += n * algorithm.update_per_rollout
            effort["ppo_sampled_transitions"] += n * algorithm.update_per_rollout * algorithm.batch_size
            effort["ppo_rollout_decisions"] += n * train.nr_paths * train.nr_steps
            stats = gradient_summary(records, clipped)
            checkpoint(name, algorithm.policy, algorithm, start + n, epoch, stats, effort)
        finals.append((name, algorithm.policy, algorithm, args.n_rounds, epoch, stats, effort.copy()))

    if args.final_test:
        print("All training finished; generating untouched final-test scenarios...", flush=True)
        split = "final_test"
        test = make_mdp(args.stage, args.n_test_paths, args.n_jobs, args.final_test_seed,
                        args.test_distribution)
        greedy = evaluate(test, GreedyPolicy(test))["costs"]
        oracle = optimal_costs(test)
        for final in finals:
            checkpoint(*final)
    else:
        print("Final test untouched. Add --final-test when the training setup is finalised.")

    if not args.no_plots:
        for name in experiments:
            selected = [r for r in rows if r["experiment"] == name and r["split"] == "validation"]
            method_dir = output_dir / name
            method_dir.mkdir(exist_ok=True)
            plot_history(selected, method_dir)
        if args.experiment:
            plot_comparison(rows, output_dir)
    print(f"Diagnostics saved to {csv_path}")
    if args.stage == 2:
        print("Note: the offline optimum can require idling, which this MDP forbids.")


if __name__ == "__main__":
    main()

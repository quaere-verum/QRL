from __future__ import annotations
import torch
from dataclasses import dataclass
from .functions import Policy
from abc import abstractmethod
from scipy.stats import norm

@dataclass
class MDP:
    action_lb: torch.Tensor
    action_ub: torch.Tensor

    def __post_init__(self):
        assert self.action_lb.ndim == self.action_ub.ndim == 1
        assert self.action_lb.shape[0] == self.action_ub.shape[0] == self.action_dim
        assert torch.all(self.action_lb < self.action_ub)
    
    @property
    def state_dim(self) -> int:
        raise NotImplementedError
    
    @property
    def action_dim(self) -> int:
        raise NotImplementedError
    
    @property
    def nr_paths(self) -> int:
        raise NotImplementedError
    
    @property
    def nr_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, policy: Policy) -> torch.Tensor:
        rewards = []
        actions = []
        state = self.reset()
        with torch.no_grad():
            for t in range(self.nr_steps):
                action = policy.act(state)
                state, reward = self.step(action)
                actions.append(action)
                rewards.append(reward)
        return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)

@dataclass
class LQGMDP(MDP):
    noise_paths: torch.Tensor
    state_transform: dict[int, torch.Tensor]
    control_transform: dict[int, torch.Tensor]
    state_penalty: dict[int, torch.Tensor]
    final_state_penalty: torch.Tensor
    control_penalty: dict[int, torch.Tensor]
    initial_state: torch.Tensor

    def __post_init__(self):
        super().__post_init__()
        for obj in [self.state_transform, self.control_transform, self.state_penalty, self.control_penalty]:
            assert isinstance(obj, dict)
            assert len(obj) == self.nr_steps
            for key in obj.keys(): 
                assert isinstance(key, int)
        for obj in [self.state_transform, self.control_transform, self.state_penalty, self.control_penalty]:
            for time in obj:
                assert obj[time].shape == obj[0].shape
        
        assert self.state_transform[0].ndim == self.control_transform[0].ndim == 2
        assert self.state_transform[0].shape[0] == self.state_transform[0].shape[1]
        assert self.control_transform[0].shape[0] == self.state_transform[0].shape[0]
        assert self.state_transform[0].shape == self.state_penalty[0].shape == self.final_state_penalty.shape
        assert self.control_penalty[0].shape[0] == self.control_penalty[0].shape[1] == self.control_transform[0].shape[1]
        assert self.initial_state.ndim == 1 and self.initial_state.shape[0] == self.state_transform[0].shape[0]
        assert self.noise_paths.ndim == 3

        self._state = torch.zeros((self.nr_paths, self.initial_state.shape[0]), dtype=torch.float32)        

    @property
    def nr_paths(self) -> int:
        return self.noise_paths.shape[0]
    
    @property
    def nr_steps(self) -> int:
        return self.noise_paths.shape[1]

    @property
    def state_dim(self) -> int:
        return self.state_transform[0].shape[0] + 1
    
    @property
    def action_dim(self) -> int:
        return self.control_transform[0].shape[1]
    
    def _get_obs(self) -> torch.Tensor:
        time_remaining = torch.full((self.nr_paths, 1), self._t / self.nr_steps, dtype=torch.float32)
        return torch.concat((self._state, time_remaining), dim=1)
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action.clamp_(self.action_lb, self.action_ub)
        if self._t >= self.nr_steps:
            raise ValueError
        
        self._t += 1
        
        cost = (
            torch.einsum(
                "bi, ij, bj -> b", self._state, self.state_penalty[self._t - 1], self._state
            ) + 
            torch.einsum(
                "bi, ij, bj -> b", action, self.control_penalty[self._t - 1], action
            )
        )

        self._state = (
            torch.matmul(self._state, self.state_transform[self._t - 1].T) +
            torch.matmul(action, self.control_transform[self._t - 1].T) +
            self.noise_paths[:, self._t - 1, :]
        )
        if self._t == self.nr_steps:
            cost += torch.einsum(
                "bi, ij, bj -> b", self._state, self.final_state_penalty, self._state
            )

        return self._get_obs(), -cost

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._state = torch.zeros((self.nr_paths, self.initial_state.shape[0]), dtype=torch.float32)
        self._state += self.initial_state
        return self._get_obs()
    
    def solve_riccati_equation(self) -> dict[int, torch.Tensor]:
        solutions = [self.final_state_penalty.clone()]
        for t in reversed(range(self.nr_steps)):
            solution = (
                self.state_transform[t].T @ (
                    solutions[-1] 
                    - solutions[-1] @ self.control_transform[t] @ torch.linalg.inv(
                        self.control_transform[t].T @ solutions[-1] @ self.control_transform[t]
                        + self.control_penalty[t]
                    ) @ self.control_transform[t].T @ solutions[-1]
                ) @ self.state_transform[t] + self.state_penalty[t]
            )
            solutions.append(solution)
        solutions = dict(zip(range(len(solutions)), solutions[::-1]))
        return solutions
    
    def solve(self) -> tuple[torch.Tensor, torch.Tensor]:
        actions, rewards = [], []
        riccati_solution = self.solve_riccati_equation()
        state = self.reset()[:, :-1]
        for t in range(self.nr_steps):
            feedback_gain_matrix = torch.linalg.inv(
                self.control_transform[t].T @ riccati_solution[t + 1] @ self.control_transform[t]
                + self.control_penalty[t]
            ) @ self.control_transform[t].T @ riccati_solution[t + 1] @ self.state_transform[t]
            action = (-torch.matmul(state, feedback_gain_matrix.T)).clamp(self.action_lb, self.action_ub)
            state, reward = self.step(action)
            state = state[:, :-1]
            actions.append(action)
            rewards.append(reward)
        return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)


@dataclass
class TradeExecutionMDP(MDP):
    """
    Finite-horizon optimal trade execution MDP (Almgren-Chriss style), optionally nonlinear.

    State per path: [inventory q_t, midprice p_t, time_remaining]
    Action per path: trade rate u_t (shares traded in this step)

    Dynamics:
        q_{t+1} = q_t - u_t
        p_{t+1} = p_t + sigma_t * eps_t - eta_t * g(u_t)

    Cost:
        c_t = lambda_t * q_t^2 + temp_cost(u_t)
        c_T = kappa * q_T^2

    Reward returned is -cost.

    Nonlinear toggles:
      - impact_fn: "linear" | "tanh" | "power"
      - temp_cost_fn: "quadratic" | "power"
    """

    noise_paths: torch.Tensor
    initial_inventory: float = 1_000.0
    initial_price: float = 100.0

    sigma: float | torch.Tensor = 0.50
    eta: float | torch.Tensor = 1e-4
    lam: float | torch.Tensor = 1e-6
    gamma: float | torch.Tensor = 1e-6
    kappa: float = 1e-4

    # ---------- Nonlinear toggles ----------
    impact_fn: str = "linear"          # "linear" | "tanh" | "power"
    temp_cost_fn: str = "quadratic"    # "quadratic" | "power"

    # impact_fn params
    impact_u_scale: float = 1_000.0    # for tanh: g(u)=u_scale*tanh(u/u_scale) ~ u near 0
    impact_power: float = 1.3          # for power: g(u)=sign(u)*(|u|+eps)^impact_power
    impact_eps: float = 1e-3           # smoothing for power

    # temp_cost_fn params
    temp_power_delta: float = 0.5      # power cost exponent is 1+delta (>1 convex)

    def __post_init__(self):
        super().__post_init__()
        assert self.action_dim == 1, "Assumes action_dim=1."
        assert self.noise_paths.ndim == 2
        self.noise_paths = self.noise_paths.to(dtype=torch.float32)

        self._sigma_t = self._to_time_series(self.sigma, "sigma")
        self._eta_t = self._to_time_series(self.eta, "eta")
        self._lam_t = self._to_time_series(self.lam, "lam")
        self._gamma_t = self._to_time_series(self.gamma, "gamma")

        self._state = torch.zeros((self.nr_paths, self.state_dim), dtype=torch.float32)

        self.action_lb = self.action_lb.to(dtype=torch.float32)
        self.action_ub = self.action_ub.to(dtype=torch.float32)

        assert self.impact_fn in ("linear", "tanh", "power")
        assert self.temp_cost_fn in ("quadratic", "power")
        if self.impact_fn == "tanh":
            assert self.impact_u_scale > 0
        if self.impact_fn == "power":
            assert self.impact_power > 1.0, "Use >1 for convex/nonlinear impact."
            assert self.impact_eps > 0
        if self.temp_cost_fn == "power":
            assert self.temp_power_delta > 0.0, "delta>0 makes temporary cost convex."

    def _to_time_series(self, x: float | torch.Tensor, name: str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return x.repeat(self.nr_steps).to(dtype=torch.float32)
            assert x.shape == (self.nr_steps,), f"{name} must be scalar or shape (nr_steps,)"
            return x.to(dtype=torch.float32)
        return torch.full((self.nr_steps,), float(x), dtype=torch.float32)

    @property
    def state_dim(self) -> int:
        return 3  # [q, p, time_remaining]

    @property
    def action_dim(self) -> int:
        return int(self.action_lb.numel())

    @property
    def nr_paths(self) -> int:
        return self.noise_paths.shape[0]

    @property
    def nr_steps(self) -> int:
        return self.noise_paths.shape[-1]

    def reset(self) -> torch.Tensor:
        self._t = 0
        q0 = torch.full((self.nr_paths, 1), float(self.initial_inventory), dtype=torch.float32)
        p0 = torch.full((self.nr_paths, 1), float(self.initial_price), dtype=torch.float32)
        time_remaining = torch.full((self.nr_paths, 1), 1.0, dtype=torch.float32)
        self._state = torch.cat([q0, p0, time_remaining], dim=1)
        return self._state

    def _g_impact(self, u: torch.Tensor) -> torch.Tensor:
        """Impact transform g(u) used in price dynamics."""
        if self.impact_fn == "linear":
            return u
        if self.impact_fn == "tanh":
            s = float(self.impact_u_scale)
            return s * torch.tanh(u / s)
        # "power" (odd, smooth)
        a = float(self.impact_power)
        eps = float(self.impact_eps)
        return torch.sign(u) * (torch.abs(u) + eps) ** a

    def _g_impact_prime(self, u: torch.Tensor) -> torch.Tensor:
        """Derivative g'(u) for linearisation."""
        if self.impact_fn == "linear":
            return torch.ones_like(u)
        if self.impact_fn == "tanh":
            s = float(self.impact_u_scale)
            # d/du [s*tanh(u/s)] = sech^2(u/s)
            x = u / s
            return 1.0 - torch.tanh(x) ** 2
        # power: d/du sign(u)*(|u|+eps)^a = a*(|u|+eps)^(a-1)
        a = float(self.impact_power)
        eps = float(self.impact_eps)
        return a * (torch.abs(u) + eps) ** (a - 1.0)

    def _temp_cost(self, u: torch.Tensor, gamma_t: torch.Tensor) -> torch.Tensor:
        """Temporary cost term (per-path), returns shape (nr_paths,)."""
        uu = u.squeeze(1)
        if self.temp_cost_fn == "quadratic":
            return gamma_t * (uu ** 2)
        delta = float(self.temp_power_delta)
        return gamma_t * (torch.abs(uu) ** (1.0 + delta))

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._t >= self.nr_steps:
            raise ValueError("Episode finished; call reset().")
        assert action.shape == (self.nr_paths, 1), f"Expected ({self.nr_paths}, 1), got {tuple(action.shape)}"

        u = action.clamp(self.action_lb, self.action_ub)

        q = self._state[:, 0:1]
        p = self._state[:, 1:2]

        sigma_t = self._sigma_t[self._t]
        eta_t = self._eta_t[self._t]
        lam_t = self._lam_t[self._t]
        gamma_t = self._gamma_t[self._t]

        stage_cost = lam_t * (q.squeeze(1) ** 2) + self._temp_cost(u, gamma_t)

        eps = self.noise_paths[:, self._t].unsqueeze(1)  # (nr_paths, 1)
        q_next = q - u

        impact_term = self._g_impact(u)  # (nr_paths,1)
        p_next = p + sigma_t * eps - eta_t * impact_term

        time_remaining_next = torch.full((self.nr_paths, 1), 1.0 - (self._t + 1) / self.nr_steps)

        self._state = torch.cat([q_next, p_next, time_remaining_next], dim=1)
        self._t += 1

        if self._t == self.nr_steps:
            terminal_cost = self.kappa * (q_next.squeeze(1) ** 2)
            stage_cost = stage_cost + terminal_cost

        reward = -stage_cost
        return self._state, reward

    def _B_t_linearised(self, t: int, u_lin: float = 0.0) -> torch.Tensor:
        """
        Effective linearised B_t for dynamics around action u_lin.
        For the price component: p_{t+1} ≈ p_t - eta_t * g'(u_lin) * u_t + const
        We ignore the affine constant in Riccati baseline (standard certainty-equivalent approximation).
        """
        eta_t = float(self._eta_t[t])
        u0 = torch.tensor([[float(u_lin)]], dtype=torch.float32)
        gprime = float(self._g_impact_prime(u0).item())
        return torch.tensor([[-1.0], [-eta_t * gprime]], dtype=torch.float32)

    def solve_riccati_equation(self, u_lin: float = 0.0) -> dict[int, torch.Tensor]:
        """
        Finite-horizon Riccati recursion for the *linearised* execution MDP:
            x_{t+1} = A x_t + B_t u_t + w_t
            cost_t  = x_t^T Q_t x_t + u_t^T R_t u_t
        with x=[q,p], u scalar.

        When impact_fn="linear" this matches your original exactly.
        When impact_fn is nonlinear, B_t uses g'(u_lin) (default u_lin=0).
        """
        T = self.nr_steps

        P = torch.zeros(2, 2, dtype=torch.float32)
        P[0, 0] = float(self.kappa)

        solutions: list[torch.Tensor] = [P]  # [P_T, P_{T-1}, ..., P_0]

        A = torch.eye(2, dtype=torch.float32)

        for t in reversed(range(T)):
            Q = torch.zeros(2, 2, dtype=torch.float32)
            Q[0, 0] = float(self._lam_t[t])

            R = torch.tensor([[float(self._gamma_t[t])]], dtype=torch.float32)

            B = self._B_t_linearised(t, u_lin=u_lin)  # (2,1)

            P_next = solutions[-1]
            S = R + B.T @ P_next @ B
            K = torch.linalg.solve(S, (B.T @ P_next @ A))
            P_t = Q + A.T @ P_next @ A - (A.T @ P_next @ B @ K)
            P_t = 0.5 * (P_t + P_t.T)
            solutions.append(P_t)

        solutions = solutions[::-1]
        return {t: solutions[t] for t in range(T + 1)}

    def solve(self, u_lin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deploy the (linear / linearised) Riccati controller inside *this* environment.
        This is your "analytical baseline" even when the environment is nonlinear.
        """
        P = self.solve_riccati_equation(u_lin=u_lin)
        state = self.reset()[:, :-1]  # [q,p]

        actions: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        A = torch.eye(2, dtype=torch.float32)

        for t in range(self.nr_steps):
            B = self._B_t_linearised(t, u_lin=u_lin)  # (2,1)
            R = torch.tensor([[float(self._gamma_t[t])]], dtype=torch.float32)

            P_next = P[t + 1]
            S = R + B.T @ P_next @ B
            K = torch.linalg.solve(S, (B.T @ P_next @ A))  # (1,2)

            u = -(state @ K.T)  # (nr_paths,1)
            u = u.clamp(self.action_lb, self.action_ub)

            state, r = self.step(u)
            state = state[:, :-1]
            actions.append(u)
            rewards.append(r)

        return torch.stack(actions, dim=1), torch.stack(rewards, dim=1)


@dataclass
class SchedulingMDP(MDP):
    arrival_times: torch.Tensor       # (B, N), nonnegative and sorted
    service_times: torch.Tensor       # (B, N, S), actual durations, strictly positive
    runtime_estimates: torch.Tensor   # (B, N, S), agent-visible estimates
    n_candidates: int = 8
    job_features: torch.Tensor | None = None  # (B, N, F), agent-visible
    history_length: int = 8

    def __post_init__(self):
        super().__post_init__()
        assert self.arrival_times.ndim == 2
        assert self.service_times.ndim == self.runtime_estimates.ndim == 3
        B, N = self.arrival_times.shape
        assert B > 0 and N > 0
        assert self.service_times.shape[:2] == (B, N)
        assert self.runtime_estimates.shape == self.service_times.shape
        assert self.n_servers > 0 and self.n_candidates > 0
        assert self.history_length >= 0
        assert torch.isfinite(self.arrival_times).all()
        assert torch.all(self.arrival_times >= 0)
        assert torch.all(self.arrival_times[:, 1:] >= self.arrival_times[:, :-1])
        assert torch.isfinite(self.service_times).all() and torch.all(self.service_times > 0)
        assert torch.isfinite(self.runtime_estimates).all() and torch.all(self.runtime_estimates > 0)
        assert self.action_lb.shape == self.action_ub.shape == (1,)
        assert self.action_lb.item() == 0 and self.action_ub.item() == self.n_actions

        if self.job_features is None:
            self.job_features = self.runtime_estimates.new_zeros((B, N, 0))
        assert self.job_features.ndim == 3 and self.job_features.shape[:2] == (B, N)
        assert torch.isfinite(self.job_features).all()

        # Simulator time is float64 even though observations remain float32.
        self.arrival_times = self.arrival_times.to(dtype=torch.float64)
        self.service_times = self.service_times.to(dtype=torch.float64)
        self.runtime_estimates = self.runtime_estimates.to(dtype=torch.float32)
        self.job_features = self.job_features.to(dtype=torch.float32)
        assert (self.arrival_times.device == self.service_times.device
                == self.runtime_estimates.device == self.job_features.device)
        self.reset()

    @property
    def nr_paths(self) -> int:
        return self.arrival_times.shape[0]

    @property
    def nr_steps(self) -> int:
        return self.arrival_times.shape[1]

    @property
    def n_servers(self) -> int:
        return self.service_times.shape[2]

    @property
    def n_actions(self) -> int:
        return self.n_candidates * self.n_servers

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def state_dim(self) -> int:
        # global(5), candidates(K*(2+S+F)), servers(S*3),
        # recent completions(H*5), action mask(K*S).
        return (5 + self.n_candidates * (2 + self.n_servers + self.job_features.shape[2])
                + 3 * self.n_servers + 5 * self.history_length + self.n_actions)

    def reset(self) -> torch.Tensor:
        B, N, S = self.nr_paths, self.nr_steps, self.n_servers
        device = self.arrival_times.device
        self._t = 0
        self._time = torch.zeros(B, dtype=torch.float64, device=device)
        self._assigned = torch.zeros((B, N), dtype=torch.bool, device=device)
        self._completed_count = torch.zeros(B, dtype=torch.long, device=device)
        self._completion_times = torch.full((B, N), float('nan'), dtype=torch.float64, device=device)
        self._running = torch.full((B, S), -1, dtype=torch.long, device=device)
        self._started = torch.zeros((B, S), dtype=torch.float64, device=device)
        self._finishes = torch.full((B, S), float('inf'), dtype=torch.float64, device=device)
        self._history = torch.zeros((B, self.history_length, 4), dtype=torch.float32, device=device)
        self._history_valid = torch.zeros((B, self.history_length), dtype=torch.bool, device=device)
        self._advance(drain=False)
        return self._get_obs()

    def _arrived_count(self) -> torch.Tensor:
        return torch.searchsorted(
            self.arrival_times.contiguous(), self._time[:, None].contiguous(), right=True
        ).squeeze(1)

    def _candidates_and_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, S = self.nr_paths, self.nr_steps, self.n_servers
        ids = torch.arange(N, device=self._time.device).expand(B, N)
        arrived = ids < self._arrived_count()[:, None]
        eligible = arrived & ~self._assigned
        pool = torch.where(eligible, ids, N)
        k = min(self.n_candidates, N)
        candidates = pool.topk(k, dim=1, largest=False, sorted=True).values
        if k < self.n_candidates:
            pad = torch.full((B, self.n_candidates - k), N, dtype=torch.long,
                             device=self._time.device)
            candidates = torch.cat((candidates, pad), dim=1)
        valid = candidates < N
        idle = self._running < 0
        mask = (valid[:, :, None] & idle[:, None, :]).reshape(B, self.n_actions)
        return candidates, mask

    def _record_finished(self, due: torch.Tensor) -> None:
        B, S = due.shape
        batch_ids = torch.arange(B, device=self._time.device)
        for s in range(S):  # loop over servers, never over Monte Carlo paths
            rows = batch_ids[due[:, s]]
            if rows.numel() == 0:
                continue
            jobs = self._running[rows, s]
            self._completion_times[rows, jobs] = self._time[rows]
            if self.history_length:
                self._history[rows, 1:] = self._history[rows, :-1].clone()
                self._history_valid[rows, 1:] = self._history_valid[rows, :-1].clone()
                self._history[rows, 0, 0] = self.runtime_estimates[rows, jobs, s]
                self._history[rows, 0, 1] = self.service_times[rows, jobs, s].float()
                self._history[rows, 0, 2] = s / max(S - 1, 1)
                self._history[rows, 0, 3] = self._time[rows].float()
                self._history_valid[rows, 0] = True
        self._completed_count += due.sum(dim=1)
        self._running[due] = -1
        self._finishes[due] = float('inf')

    def _advance(self, *, drain: bool) -> torch.Tensor:
        """Advance each path to its next decision, or drain after last dispatch.

        Returns integrated unfinished-job count on each path (float64).
        """
        B, N = self.nr_paths, self.nr_steps
        device = self._time.device
        cost = torch.zeros(B, dtype=torch.float64, device=device)
        inf = torch.full((B,), float('inf'), dtype=torch.float64, device=device)
        batch_ids = torch.arange(B, device=device)

        # At most N arrivals and N completions exist on each path.
        for _ in range(2 * N + 1):
            arrived = self._arrived_count()
            if drain:
                need = (self._running >= 0).any(dim=1)
            else:
                waiting = (arrived - self._assigned.sum(dim=1)) > 0
                need = ~(waiting & (self._running < 0).any(dim=1))
            if not need.any():
                return cost

            next_arrival = torch.where(
                arrived < N,
                self.arrival_times[batch_ids, arrived.clamp(max=N - 1)],
                inf,
            )
            if drain:
                next_arrival = inf
            next_completion = self._finishes.min(dim=1).values
            next_time = torch.minimum(next_arrival, next_completion)
            if not torch.isfinite(next_time[need]).all():
                raise RuntimeError('Scheduling deadlock: no future observable events')
            next_time = torch.where(need, next_time, self._time)
            dt = next_time - self._time
            if torch.any(dt < 0):
                raise RuntimeError('Event time went backwards')
            cost += (arrived - self._completed_count).to(torch.float64) * dt
            self._time = next_time
            due = (self._running >= 0) & (self._finishes <= self._time[:, None])
            if due.any():
                self._record_finished(due)

        raise RuntimeError('Exceeded maximum event count')

    def _get_obs(self) -> torch.Tensor:
        B, N, S, K = self.nr_paths, self.nr_steps, self.n_servers, self.n_candidates
        candidates, mask = self._candidates_and_mask()
        valid = candidates < N
        ids = candidates.clamp(max=N - 1)
        rows = torch.arange(B, device=self._time.device)[:, None]

        ages = torch.where(valid, (self._time[:, None] - self.arrival_times[rows, ids]).float(), 0.)
        estimates = self.runtime_estimates[rows, ids] * valid[:, :, None]
        features = self.job_features[rows, ids] * valid[:, :, None]
        candidate_obs = torch.cat((valid.float()[:, :, None], ages[:, :, None],
                                   estimates, features), dim=-1).reshape(B, -1)

        running = self._running >= 0
        running_ids = self._running.clamp(min=0)
        server_ids = torch.arange(S, device=self._time.device)[None, :]
        running_estimates = self.runtime_estimates[rows, running_ids, server_ids]
        server_obs = torch.stack((
            (~running).float(),
            torch.where(running, (self._time[:, None] - self._started).float(), 0.),
            torch.where(running, running_estimates, 0.),
        ), dim=-1).reshape(B, -1)

        history = self._history.clone()
        if self.history_length:
            history[:, :, 3] = torch.where(
                self._history_valid,
                self._time[:, None].float() - history[:, :, 3],
                0.,
            )
        history_obs = torch.cat((self._history_valid.float()[:, :, None], history),
                                dim=-1).reshape(B, -1)

        arrived = self._arrived_count()
        global_obs = torch.stack((
            self._time.float(),
            torch.full((B,), self._t / N, device=self._time.device),
            arrived.float() / N,
            self._completed_count.float() / N,
            (arrived - self._assigned.sum(dim=1)).float() / N,
        ), dim=1)
        return torch.cat((global_obs, candidate_obs, server_obs, history_obs,
                          mask.float()), dim=1)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._t >= self.nr_steps:
            raise ValueError('Episode finished; call reset()')
        if action.shape != (self.nr_paths, 1) or action.dtype not in (torch.int32, torch.int64):
            raise ValueError('Discrete actions must have integer shape (nr_paths, 1)')
        if torch.any((action < self.action_lb) | (action >= self.action_ub)):
            raise ValueError('Action outside [0, n_actions)')

        candidates, mask = self._candidates_and_mask()
        selection = action[:, 0].long()
        batch_ids = torch.arange(self.nr_paths, device=self._time.device)
        if not mask[batch_ids, selection].all():
            raise ValueError('Illegal assignment: absent job or busy server')

        slots = torch.div(selection, self.n_servers, rounding_mode='floor')
        servers = selection % self.n_servers
        jobs = candidates[batch_ids, slots]
        self._assigned[batch_ids, jobs] = True
        self._running[batch_ids, servers] = jobs
        self._started[batch_ids, servers] = self._time
        self._finishes[batch_ids, servers] = self._time + self.service_times[batch_ids, jobs, servers]
        self._t += 1
        holding_cost = self._advance(drain=(self._t == self.nr_steps))
        return self._get_obs(), (-holding_cost / self.nr_steps).float()

    def mean_sojourn(self) -> torch.Tensor:
        if self._t != self.nr_steps or not (self._completed_count == self.nr_steps).all():
            raise ValueError('The episode has not completed')
        return (self._completion_times - self.arrival_times).mean(dim=1)

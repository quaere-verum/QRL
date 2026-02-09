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

    # -----------------------
    # Nonlinear building blocks
    # -----------------------
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
        # convex power: gamma * |u|^{1+delta}
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

        # cost at current state/action
        stage_cost = lam_t * (q.squeeze(1) ** 2) + self._temp_cost(u, gamma_t)

        # dynamics
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

    # -----------------------
    # Linear baseline (your existing Riccati) + optional linearised B
    # -----------------------
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

            # NOTE: if you also want a quadratic approximation of the convex temp cost,
            # you can replace R below by a local curvature term around u_lin.
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

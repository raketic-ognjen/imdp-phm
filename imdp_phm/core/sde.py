"""Coupled Endogenous Hawkes-Merton Jump-Diffusion SDE Simulator & Generator."""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from .params import HawkesMertonParams, StructuralConstants


@dataclass
class DiscreteSimulationResult:
    """Output of vectorized macro-epoch first-exit simulation (RESS Algorithm 1)."""
    time_grid: np.ndarray          # (K_max + 1,) [h]
    h_paths: np.ndarray            # (N, K_max + 1) damage trajectories
    lambda_paths: np.ndarray       # (N, K_max + 1) conditional intensity trajectories
    jump_counts: np.ndarray        # (N, K_max) binary/integer shock indicators per epoch
    tau_major: np.ndarray          # (N,) persistent alert epoch [h] (NaN if unalerted)
    tau_crit: np.ndarray           # (N,) absorbing barrier first-exit epoch [h] (NaN if censored)
    lead_to_imminent_hazard: np.ndarray  # (N,) Delta tau_LIH = tau_crit - tau_major [h]


@dataclass
class ContinuousTrajectoryResult:
    """Output of exact continuous-time Ogata thinning simulation for point-process diagnostics."""
    t_grid: np.ndarray             # Fine evaluation grid [h]
    h_grid: np.ndarray             # Damage H(t) on grid
    lambda_grid: np.ndarray        # Intensity lambda(t) on grid
    jump_times: np.ndarray         # Exact event arrival times {t_j} [h]
    jump_sizes: np.ndarray         # Log-normal shock sizes {Y_j} [-]
    h_at_jumps: np.ndarray         # Pre-jump damage states {H(t_j-)} [-]
    tau_crit: float                # Exact first-exit time (np.inf if not reached before t_max)


class CoupledHawkesMertonSDE:
    """Bivariate Markovian degradation model Z_t = (H_t, lambda_t) (Dion et al., 2021; Raketic, Do & Zio)."""

    def __init__(
        self,
        params: Optional[HawkesMertonParams] = None,
        constants: Optional[StructuralConstants] = None,
    ) -> None:
        self.params = params or HawkesMertonParams()
        self.constants = constants or StructuralConstants()
        self.params.validate()

    def drift(self, h: np.ndarray | float) -> np.ndarray | float:
        """State-dependent continuous wear drift a(H) (RESS Eq. 2)."""
        h_norm = np.clip(h, 0.0, self.constants.h_crit) / self.constants.h_crit
        return self.params.a0 * (1.0 + self.constants.zeta_a * (h_norm ** self.constants.theta_a))

    def volatility(self, h: np.ndarray | float) -> np.ndarray | float:
        """State-dependent Brownian diffusion volatility sigma(H) (RESS Eq. 3)."""
        h_norm = np.clip(h, 0.0, self.constants.h_crit) / self.constants.h_crit
        return self.params.sigma0 * np.sqrt(1.0 + self.constants.zeta_sigma * h_norm)

    def excitation_alpha(self, h: np.ndarray | float) -> np.ndarray | float:
        """Quadratic state-dependent structural excitation alpha(H) (RESS Eq. 7)."""
        h_capped = np.minimum(np.maximum(h, 0.0), self.constants.h_crit) / self.constants.h_crit
        return self.params.alpha0 * (1.0 + self.params.gamma * (h_capped ** 2))

    def generator(
        self,
        h: float,
        lam: float,
        f: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float],
        eps_h: float = 1e-5,
        eps_lam: float = 1e-5,
        n_quad: int = 32,
    ) -> float:
        """Infinitesimal generator L_theta f(h, lambda) (Concept Note Eq. 3) via Gauss-Hermite quadrature."""
        df_dh = (f(h + eps_h, lam) - f(h - eps_h, lam)) / (2.0 * eps_h)
        d2f_dh2 = (f(h + eps_h, lam) - 2.0 * f(h, lam) + f(h - eps_h, lam)) / (eps_h ** 2)
        df_dlam = (f(h, lam + eps_lam) - f(h, lam - eps_lam)) / (2.0 * eps_lam)

        # Gauss-Hermite quadrature for E_Y[f(h + Y, lam + alpha(h)) - f(h, lam)], ln(Y) ~ N(mu_J, sigma_J^2)
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        y_vals = np.exp(self.params.mu_J + np.sqrt(2.0) * self.params.sigma_J * nodes)
        alpha_h = float(self.excitation_alpha(h))
        jump_integrand = f(h + y_vals, lam + alpha_h) - f(h, lam)
        jump_expectation = float(np.sum(weights * jump_integrand) / np.sqrt(np.pi))

        a_h = float(self.drift(h))
        sig_h = float(self.volatility(h))
        return (
            a_h * df_dh
            + 0.5 * (sig_h ** 2) * d2f_dh2
            + self.params.beta * (self.params.mu0 - lam) * df_dlam
            + lam * jump_expectation
        )

    def simulate_macro_epochs(
        self,
        n_paths: int = 1000,
        t_max: float = 200.0,
        seed: int = 0x8F5C2A4D,
        h_init: Optional[float] = None,
        lam_init: Optional[float] = None,
    ) -> DiscreteSimulationResult:
        """Vectorized macro-epoch Euler-Maruyama + O(1) recursive Hawkes simulation (RESS Algorithm 1)."""
        rng = np.random.Generator(np.random.PCG64(seed))
        dt = self.constants.dt
        h_crit = self.constants.h_crit
        lam_th = self.constants.lambda_th
        k_max = int(np.floor(t_max / dt))

        h0_val = self.constants.h0 if h_init is None else float(h_init)
        lam0_val = self.params.mu0 if lam_init is None else float(lam_init)

        h = np.full(n_paths, h0_val, dtype=np.float64)
        lam = np.full(n_paths, lam0_val, dtype=np.float64)
        lam_prev = lam.copy()
        active = np.ones(n_paths, dtype=bool)
        rho = float(np.exp(-self.params.beta * dt))

        tau_major = np.full(n_paths, np.nan, dtype=np.float64)
        tau_crit = np.full(n_paths, np.nan, dtype=np.float64)

        time_grid = np.arange(k_max + 1, dtype=np.float64) * dt
        h_paths = np.empty((n_paths, k_max + 1), dtype=np.float64)
        lambda_paths = np.empty((n_paths, k_max + 1), dtype=np.float64)
        jump_counts = np.zeros((n_paths, k_max), dtype=np.int32)

        h_paths[:, 0] = h
        lambda_paths[:, 0] = lam
        sqrt_dt = np.sqrt(dt)

        for k in range(1, k_max + 1):
            n_act = int(np.sum(active))
            if n_act == 0:
                h_paths[:, k:] = h[:, None]
                lambda_paths[:, k:] = lam[:, None]
                break

            t_k = k * dt
            h_act = h[active]
            lam_act = lam[active]

            # 1. Continuous SDE drift and diffusion
            a_step = self.drift(h_act) * dt
            sig_step = self.volatility(h_act)
            dw = sqrt_dt * rng.standard_normal(n_act)

            # 2. Jump occurrence via Poisson thinning over epoch Delta t
            p_jump = 1.0 - np.exp(-lam_act * dt)
            jumps = rng.random(n_act) < p_jump

            # 3. Sample log-normal jump severity
            y_jump = np.zeros(n_act, dtype=np.float64)
            n_jumps = int(np.sum(jumps))
            if n_jumps > 0:
                y_jump[jumps] = np.exp(
                    self.params.mu_J + self.params.sigma_J * rng.standard_normal(n_jumps)
                )

            # 4. Update state H_t and O(1) recursive hazard intensity lambda_t
            h_next = np.maximum(0.0, h_act + a_step + sig_step * dw + y_jump)
            alpha_t = self.excitation_alpha(h_act)
            lam_next = self.params.mu0 + (lam_act - self.params.mu0) * rho + jumps * alpha_t

            h[active] = h_next
            lam[active] = lam_next
            jump_counts[active, k - 1] = jumps.astype(np.int32)

            h_paths[:, k] = h
            lambda_paths[:, k] = lam

            # 5. Persistent alarm and absorbing barrier checks
            alert_mask = active & np.isnan(tau_major) & (lam >= lam_th) & (lam_prev >= lam_th)
            tau_major[alert_mask] = t_k

            fail_mask = active & (h >= h_crit)
            tau_crit[fail_mask] = t_k
            active[fail_mask] = False
            lam_prev = lam.copy()

        lih = tau_crit - tau_major
        return DiscreteSimulationResult(
            time_grid=time_grid,
            h_paths=h_paths,
            lambda_paths=lambda_paths,
            jump_counts=jump_counts,
            tau_major=tau_major,
            tau_crit=tau_crit,
            lead_to_imminent_hazard=lih,
        )

    def simulate_continuous_ogata(
        self,
        t_max: float = 150.0,
        dt_sub: float = 0.01,
        seed: int = 42,
        h_init: Optional[float] = None,
        lam_init: Optional[float] = None,
        stop_at_crit: bool = True,
    ) -> ContinuousTrajectoryResult:
        """Exact continuous-time Ogata thinning + fine Euler-Maruyama trajectory for point-process tests."""
        rng = np.random.Generator(np.random.PCG64(seed))
        t = 0.0
        h = self.constants.h0 if h_init is None else float(h_init)
        lam = self.params.mu0 if lam_init is None else float(lam_init)

        t_hist = [t]
        h_hist = [h]
        lam_hist = [lam]
        jump_times = []
        jump_sizes = []
        h_at_jumps = []
        tau_crit = np.inf

        while t < t_max:
            # Upper bound M_t for thinning over [t, t + dt_sub]:
            # Since beta > 0 and mu0 > 0, lambda(s) <= max(lambda(t), mu0) until next jump
            lam_bar = max(lam, self.params.mu0)
            wait = rng.exponential(1.0 / lam_bar)
            step = min(wait, dt_sub, t_max - t)

            # Evolve continuous diffusion and deterministic intensity decay over `step`
            if step > 0:
                dw = np.sqrt(step) * rng.standard_normal()
                h_new = max(0.0, h + float(self.drift(h)) * step + float(self.volatility(h)) * dw)
                lam_new = self.params.mu0 + (lam - self.params.mu0) * np.exp(-self.params.beta * step)
                t += step
                h = h_new
                lam = lam_new

            if wait <= dt_sub and t < t_max:
                # Thinning acceptance test at candidate event time t
                if rng.random() <= lam / lam_bar:
                    y_j = float(np.exp(self.params.mu_J + self.params.sigma_J * rng.standard_normal()))
                    h_at_jumps.append(h)
                    jump_times.append(t)
                    jump_sizes.append(y_j)

                    alpha_j = float(self.excitation_alpha(h))
                    h = h + y_j
                    lam = lam + alpha_j

            t_hist.append(t)
            h_hist.append(h)
            lam_hist.append(lam)

            if stop_at_crit and h >= self.constants.h_crit:
                tau_crit = t
                break

        return ContinuousTrajectoryResult(
            t_grid=np.array(t_hist, dtype=np.float64),
            h_grid=np.array(h_hist, dtype=np.float64),
            lambda_grid=np.array(lam_hist, dtype=np.float64),
            jump_times=np.array(jump_times, dtype=np.float64),
            jump_sizes=np.array(jump_sizes, dtype=np.float64),
            h_at_jumps=np.array(h_at_jumps, dtype=np.float64),
            tau_crit=float(tau_crit),
        )

    def compensator_residuals(self, traj: ContinuousTrajectoryResult) -> np.ndarray:
        """Rescaled inter-event residuals tau_k* = Lambda(t_k) - Lambda(t_{k-1}) via Random Time Change Theorem."""
        t_j = traj.jump_times
        if len(t_j) < 2:
            return np.array([], dtype=np.float64)

        # Exact analytical integral of piecewise exponential intensity between consecutive jump times:
        # On (t_{k-1}, t_k], lambda(s) = mu0 + (lambda(t_{k-1}+) - mu0) * exp(-beta * (s - t_{k-1}))
        residuals = np.empty(len(t_j), dtype=np.float64)
        lam_plus = self.params.mu0
        t_prev = 0.0

        for k, (tk, hk) in enumerate(zip(t_j, traj.h_at_jumps)):
            dt_k = tk - t_prev
            residuals[k] = (
                self.params.mu0 * dt_k
                + ((lam_plus - self.params.mu0) / self.params.beta) * (1.0 - np.exp(-self.params.beta * dt_k))
            )
            lam_minus = self.params.mu0 + (lam_plus - self.params.mu0) * np.exp(-self.params.beta * dt_k)
            lam_plus = lam_minus + float(self.excitation_alpha(hk))
            t_prev = tk

        return residuals

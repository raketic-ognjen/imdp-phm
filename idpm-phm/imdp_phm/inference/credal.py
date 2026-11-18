"""Convex Credal Parameter Sets Theta in R^8 and Parametric P-Boxes F_J (Concept Note Sec. 2; Faes et al., 2021)."""

from dataclasses import dataclass
from itertools import product
from typing import Callable
import numpy as np
from scipy.stats import norm

from imdp_phm.inference.mle import MLEEstimationResult


PARAM_NAMES = ("a0", "sigma0", "mu0", "alpha0", "beta", "gamma", "mu_J", "sigma_J")


@dataclass(frozen=True)
class JumpSeverityPBox:
    """Parametric probability box F_J in [F_J_low, F_J_high] for log-normal jump severity
    ln(Y) ~ N(mu_J, sigma_J^2) with mu_J in [mu_low, mu_high] and sigma_J in [sigma_low, sigma_high].
    """
    mu_low: float
    mu_high: float
    sigma_low: float
    sigma_high: float

    def __post_init__(self) -> None:
        if self.mu_low > self.mu_high:
            raise ValueError("mu_low must be <= mu_high")
        if self.sigma_low <= 0.0 or self.sigma_low > self.sigma_high:
            raise ValueError("Must satisfy 0 < sigma_low <= sigma_high")

    def cdf_lower(self, y: np.ndarray | float) -> np.ndarray | float:
        """Exact pointwise lower CDF bound F_J_low(y) = inf_{(mu_J, sigma_J)} Phi((ln y - mu_J) / sigma_J)."""
        y_arr = np.asarray(y, dtype=np.float64)
        out = np.zeros_like(y_arr, dtype=np.float64)
        pos = y_arr > 0.0
        ln_y = np.log(y_arr[pos])
        c1 = norm.cdf((ln_y - self.mu_high) / self.sigma_low)
        c2 = norm.cdf((ln_y - self.mu_high) / self.sigma_high)
        out[pos] = np.minimum(c1, c2)
        return float(out) if np.ndim(y) == 0 else out

    def cdf_upper(self, y: np.ndarray | float) -> np.ndarray | float:
        """Exact pointwise upper CDF bound F_J_high(y) = sup_{(mu_J, sigma_J)} Phi((ln y - mu_J) / sigma_J)."""
        y_arr = np.asarray(y, dtype=np.float64)
        out = np.zeros_like(y_arr, dtype=np.float64)
        pos = y_arr > 0.0
        ln_y = np.log(y_arr[pos])
        c1 = norm.cdf((ln_y - self.mu_low) / self.sigma_low)
        c2 = norm.cdf((ln_y - self.mu_low) / self.sigma_high)
        out[pos] = np.maximum(c1, c2)
        return float(out) if np.ndim(y) == 0 else out

    def quantile_upper_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Inverse of lower CDF F_J_low^{-1}(u): stochastic upper-bound jump realization for u in (0, 1)."""
        u_arr = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z = norm.ppf(u_arr)
        y1 = np.exp(self.mu_high + self.sigma_low * z)
        y2 = np.exp(self.mu_high + self.sigma_high * z)
        res = np.maximum(y1, y2)
        return float(res) if np.ndim(u) == 0 else res

    def quantile_lower_bound(self, u: np.ndarray | float) -> np.ndarray | float:
        """Inverse of upper CDF F_J_high^{-1}(u): stochastic lower-bound jump realization for u in (0, 1)."""
        u_arr = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z = norm.ppf(u_arr)
        y1 = np.exp(self.mu_low + self.sigma_low * z)
        y2 = np.exp(self.mu_low + self.sigma_high * z)
        res = np.minimum(y1, y2)
        return float(res) if np.ndim(u) == 0 else res

    def expectation_bounds(
        self,
        g: Callable[[np.ndarray], np.ndarray],
        n_quad: int = 256,
    ) -> tuple[float, float]:
        """Exact lower and upper expectation bounds [E_low[g(Y)], E_high[g(Y)]]
        for any non-decreasing function g(y) via probability integral transform over [0, 1].
        """
        u_mid = (np.arange(n_quad, dtype=np.float64) + 0.5) / n_quad
        y_low = self.quantile_lower_bound(u_mid)
        y_high = self.quantile_upper_bound(u_mid)
        return float(np.mean(g(y_low))), float(np.mean(g(y_high)))


@dataclass
class CredalParameterSet:
    """Convex credal parameter set Theta subset R^8 for theta = (a0, sigma0, mu0, alpha0, beta, gamma, mu_J, sigma_J)
    enforcing physical positivity and subcritical branching ratio n_max = alpha0*(1+gamma)/beta <= max_branching_cap.
    """
    lower: np.ndarray            # (8,) lower bounds
    upper: np.ndarray            # (8,) upper bounds
    max_branching_cap: float = 0.99
    active_regime: str = "NH-Hawkes"

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        if self.lower.shape != (8,) or self.upper.shape != (8,):
            raise ValueError("CredalParameterSet bounds must have shape (8,).")
        if np.any(self.lower > self.upper):
            raise ValueError(f"Invalid credal bounds: lower > upper at {np.where(self.lower > self.upper)[0]}")
        for idx in (0, 1, 2, 4, 7):
            if self.lower[idx] <= 0.0:
                raise ValueError(f"Parameter {PARAM_NAMES[idx]} lower bound must be strictly positive.")
        if self.lower[3] < 0.0 or self.lower[5] < 0.0:
            raise ValueError("Excitation parameters (alpha0, gamma) must have lower bound >= 0.")
        min_branching = self.lower[3] * (1.0 + self.lower[5]) / self.upper[4]
        if min_branching > self.max_branching_cap:
            self.lower[3] = max(0.0, 0.95 * self.max_branching_cap * self.upper[4] / (1.0 + self.lower[5]))

    @classmethod
    def from_mle_result(
        cls,
        mle: MLEEstimationResult,
        n_trajectories: int = 1,
        z_mult: float = 1.96,
    ) -> "CredalParameterSet":
        """Construct convex credal set Theta from MLE point estimates and Fisher standard errors,
        applying a small-sample epistemic scarcity inflation factor 1 + 1/sqrt(M) (Concept Note Sec. 2).
        """
        theta_hat = mle.effective_params.to_vector()
        se = mle.standard_errors
        scarcity_factor = 1.0 + 1.0 / np.sqrt(max(int(n_trajectories), 1))
        half_width = z_mult * se * scarcity_factor

        low = theta_hat - half_width
        high = theta_hat + half_width

        pos_floors = np.array([1e-6, 1e-5, 1e-3, 0.0, 0.05, 0.0, -np.inf, 0.05], dtype=np.float64)
        low = np.maximum(low, pos_floors)
        high = np.maximum(high, low)

        if mle.hypothesis_test.active_regime == "NHPP-Fallback":
            low[3], high[3] = 0.0, 0.0
            low[5], high[5] = 0.0, 0.0

        return cls(
            lower=low,
            upper=high,
            max_branching_cap=0.99,
            active_regime=mle.hypothesis_test.active_regime,
        )

    @classmethod
    def from_bootstrap_samples(
        cls,
        samples: np.ndarray,
        q_low: float = 0.025,
        q_high: float = 0.975,
        inflation: float = 0.10,
        active_regime: str = "NH-Hawkes",
    ) -> "CredalParameterSet":
        """Construct credal box from bootstrap trajectory estimates with epistemic margin inflation."""
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError("Expected bootstrap array of shape (B, 8).")
        lo = np.quantile(arr, q_low, axis=0)
        hi = np.quantile(arr, q_high, axis=0)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * (1.0 + inflation)
        low = mid - half
        high = mid + half

        pos_floors = np.array([1e-6, 1e-5, 1e-3, 0.0, 0.05, 0.0, -np.inf, 0.05], dtype=np.float64)
        low = np.maximum(low, pos_floors)
        high = np.maximum(high, low)
        if active_regime == "NHPP-Fallback":
            low[3], high[3] = 0.0, 0.0
            low[5], high[5] = 0.0, 0.0

        return cls(lower=low, upper=high, active_regime=active_regime)

    def jump_pbox(self) -> JumpSeverityPBox:
        """Extract the induced parametric jump-severity p-box F_J in [F_J_low, F_J_high]."""
        return JumpSeverityPBox(
            mu_low=float(self.lower[6]),
            mu_high=float(self.upper[6]),
            sigma_low=float(self.lower[7]),
            sigma_high=float(self.upper[7]),
        )

    def contains(self, theta_vec: np.ndarray, tol: float = 1e-12) -> bool:
        """Check whether theta_vec belongs to convex credal set Theta (box + subcriticality constraint)."""
        v = np.asarray(theta_vec, dtype=np.float64)
        if v.shape != (8,):
            return False
        if np.any(v < self.lower - tol) or np.any(v > self.upper + tol):
            return False
        n_max = v[3] * (1.0 + v[5]) / v[4]
        return bool(n_max <= self.max_branching_cap + tol)

    def project(self, theta_vec: np.ndarray) -> np.ndarray:
        """Project candidate parameter vector onto subcritical credal set Theta."""
        v = np.clip(np.asarray(theta_vec, dtype=np.float64).copy(), self.lower, self.upper)
        if v[3] * (1.0 + v[5]) / v[4] > self.max_branching_cap and v[3] > 0.0:
            v[3] = np.clip(self.max_branching_cap * v[4] / (1.0 + v[5]), self.lower[3], self.upper[3])
        if v[3] * (1.0 + v[5]) / v[4] > self.max_branching_cap and v[3] > 0.0:
            target_gamma = (self.max_branching_cap * v[4] / v[3]) - 1.0
            v[5] = np.clip(target_gamma, self.lower[5], self.upper[5])
        if v[3] * (1.0 + v[5]) / v[4] > self.max_branching_cap and v[3] > 0.0:
            target_beta = v[3] * (1.0 + v[5]) / self.max_branching_cap
            v[4] = np.clip(target_beta, self.lower[4], self.upper[4])
        return v

    def vertices(self, project_subcritical: bool = True) -> np.ndarray:
        """Enumerate unique extreme vertices of the credal box (projected to subcritical domain if requested)."""
        axes = [
            [float(self.lower[i])] if np.isclose(self.lower[i], self.upper[i]) else [float(self.lower[i]), float(self.upper[i])]
            for i in range(8)
        ]
        raw = np.array(list(product(*axes)), dtype=np.float64)
        if project_subcritical:
            raw = np.array([self.project(row) for row in raw], dtype=np.float64)
            raw = np.unique(np.round(raw, decimals=12), axis=0)
        return raw

    def sample_latin_hypercube(self, n_samples: int = 64, seed: int = 42) -> np.ndarray:
        """Generate space-filling Latin Hypercube samples inside Theta for GP/PCE surrogate training in WP2."""
        rng = np.random.Generator(np.random.PCG64(seed))
        samples = np.empty((n_samples, 8), dtype=np.float64)
        for j in range(8):
            perm = rng.permutation(n_samples)
            u = (perm + rng.random(n_samples)) / n_samples
            samples[:, j] = self.lower[j] + u * (self.upper[j] - self.lower[j])
        return np.array([self.project(row) for row in samples], dtype=np.float64)

    def interval_summary(self) -> dict[str, tuple[float, float]]:
        """Human-readable dictionary mapping each parameter name to (lower, upper)."""
        return {name: (float(self.lower[i]), float(self.upper[i])) for i, name in enumerate(PARAM_NAMES)}

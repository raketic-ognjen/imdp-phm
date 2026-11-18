"""Parameter definitions and calibrated defaults from Raketic, Do & Zio (RESS)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StructuralConstants:
    """Fixed physical and discretization constants (RESS Sections 2.2, 3.2-3.3)."""
    h0: float = 0.10          # Nominal pristine EHD damage baseline [-]
    h_crit: float = 2.50      # Absorbing spalling saturation barrier [-]
    zeta_a: float = 1.85      # Non-linear drift acceleration scale [-]
    theta_a: float = 1.50     # Non-linear drift exponent [-]
    zeta_sigma: float = 2.10  # State-dependent diffusion scaling [-]
    dt: float = 0.1667        # Macro-epoch interval [h] (10 min = 1/6 h)
    lambda_th: float = 0.40   # Persistent hazard alert threshold [h^-1]
    kappa_d: float = 0.042    # Dimensional damage scaling [h^-1]
    z_thresh: float = 4.5     # Baseline noise screening multiplier [-]


@dataclass(frozen=True)
class HawkesMertonParams:
    """8-dimensional epistemic parameter vector theta = (a0, sigma0, mu0, alpha0, beta, gamma, mu_J, sigma_J)."""
    a0: float = 3.12e-4       # Baseline continuous wear drift [h^-1]
    sigma0: float = 1.18e-3   # Baseline Brownian volatility [h^-1/2]
    mu0: float = 0.15         # Nominal exogenous hazard rate [h^-1]
    alpha0: float = 0.35      # Baseline shock excitation sensitivity [h^-1]
    beta: float = 0.85        # Elastodynamic stress-relaxation rate [h^-1]
    gamma: float = 1.40       # Quadratic structural feedback gain [-]
    mu_J: float = -3.344      # Log-normal jump severity mean [-]
    sigma_J: float = 0.996    # Log-normal jump severity std [-]

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.a0, self.sigma0, self.mu0, self.alpha0, self.beta, self.gamma, self.mu_J, self.sigma_J],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "HawkesMertonParams":
        v = np.asarray(vec, dtype=np.float64)
        if v.shape != (8,):
            raise ValueError(f"Expected 8-element parameter vector, got shape {v.shape}")
        return cls(
            a0=float(v[0]),
            sigma0=float(v[1]),
            mu0=float(v[2]),
            alpha0=float(v[3]),
            beta=float(v[4]),
            gamma=float(v[5]),
            mu_J=float(v[6]),
            sigma_J=float(v[7]),
        )

    @classmethod
    def nasa_ims_test2_se(cls) -> np.ndarray:
        """Asymptotic standard errors from MLE / Fisher Information (RESS Section 3.4)."""
        return np.array([0.24e-4, 0.11e-3, 0.02, 0.04, 0.06, 0.20, 0.127, 0.090], dtype=np.float64)

    def nominal_branching_ratio(self) -> float:
        """Nominal branching ratio n(0) = alpha0 / beta."""
        return self.alpha0 / self.beta

    def max_branching_ratio(self) -> float:
        """Maximum saturated branching ratio n_max = alpha0 * (1 + gamma) / beta at H = H_crit."""
        return self.alpha0 * (1.0 + self.gamma) / self.beta

    def expected_jump_size(self) -> float:
        """E[Y] = exp(mu_J + 0.5 * sigma_J^2)."""
        return float(np.exp(self.mu_J + 0.5 * (self.sigma_J ** 2)))

    def validate(self) -> None:
        if self.a0 <= 0 or self.sigma0 <= 0 or self.mu0 <= 0 or self.beta <= 0 or self.sigma_J <= 0:
            raise ValueError("Scale and rate parameters (a0, sigma0, mu0, beta, sigma_J) must be strictly positive.")
        if self.alpha0 < 0 or self.gamma < 0:
            raise ValueError("Excitation parameters (alpha0, gamma) must be non-negative.")

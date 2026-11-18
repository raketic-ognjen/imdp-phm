"""2D Bivariate State Discretization (H_t, lambda_t), Action Space A, and Cost Structure (Concept Note Sec. 2)."""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class MaintenanceAction(IntEnum):
    """Discrete maintenance action space A (Concept Note Revision 3, Section 2)."""
    CONTINUE = 0               # Continue normal operation
    NDE_INSPECTION = 1         # Perform Non-Destructive Evaluation (NDE) inspection
    PREVENTIVE_REPLACEMENT = 2 # Preventive replacement resetting state to pristine (h0, mu0)


@dataclass(frozen=True)
class MaintenanceCostModel:
    """Economic cost structure c(h, lambda, a) and discount factor gamma_d in (0, 1) (Concept Note Eq. 5)."""
    c_continue_op: float = 0.0           # Nominal epoch operating cost
    c_inspection: float = 15.0           # NDE inspection cost
    c_preventive: float = 100.0          # Planned preventive replacement cost
    c_failure: float = 1200.0            # Uncontained catastrophic failure penalty upon H >= H_crit
    inspection_hazard_relief: float = 0.65  # Fraction of excess hazard (lambda - mu0) mitigated by NDE intervention
    inspection_damage_halt: float = 0.50    # Scaling factor on epoch continuous wear drift under NDE intervention
    discount_factor: float = 0.98        # Per-epoch discount factor gamma_d in (0, 1)

    def __post_init__(self) -> None:
        if not (0.0 < self.discount_factor < 1.0):
            raise ValueError("discount_factor gamma_d must lie strictly in (0, 1).")
        if self.c_inspection < 0.0 or self.c_preventive < 0.0 or self.c_failure <= self.c_preventive:
            raise ValueError("Costs must be non-negative with c_failure > c_preventive.")
        if not (0.0 <= self.inspection_hazard_relief <= 1.0):
            raise ValueError("inspection_hazard_relief must lie in [0, 1].")
        if not (0.0 <= self.inspection_damage_halt <= 1.0):
            raise ValueError("inspection_damage_halt must lie in [0, 1].")


@dataclass
class StateGrid2D:
    """Tensor-product 2D grid over S = [h_min, h_crit] x [lam_min, lam_max] with vectorized bilinear interpolation."""
    h_grid: np.ndarray    # (N_h,) ascending grid ending at h_crit
    lam_grid: np.ndarray  # (N_l,) ascending grid starting at lam_min

    def __post_init__(self) -> None:
        self.h_grid = np.asarray(self.h_grid, dtype=np.float64)
        self.lam_grid = np.asarray(self.lam_grid, dtype=np.float64)
        if self.h_grid.ndim != 1 or len(self.h_grid) < 3:
            raise ValueError("h_grid must be 1D with at least 3 nodes.")
        if self.lam_grid.ndim != 1 or len(self.lam_grid) < 3:
            raise ValueError("lam_grid must be 1D with at least 3 nodes.")
        if np.any(np.diff(self.h_grid) <= 0.0) or np.any(np.diff(self.lam_grid) <= 0.0):
            raise ValueError("Grid coordinates must be strictly ascending.")

    @classmethod
    def uniform(
        cls,
        h_min: float = 0.10,
        h_crit: float = 2.50,
        lam_min: float = 0.10,
        lam_max: float = 2.50,
        n_h: int = 41,
        n_lam: int = 31,
    ) -> "StateGrid2D":
        return cls(
            h_grid=np.linspace(h_min, h_crit, n_h, dtype=np.float64),
            lam_grid=np.linspace(lam_min, lam_max, n_lam, dtype=np.float64),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.h_grid), len(self.lam_grid))

    @property
    def h_crit(self) -> float:
        return float(self.h_grid[-1])

    @property
    def h_min(self) -> float:
        return float(self.h_grid[0])

    @property
    def lam_min(self) -> float:
        return float(self.lam_grid[0])

    @property
    def lam_max(self) -> float:
        return float(self.lam_grid[-1])

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (H_mesh, Lam_mesh) each of shape (N_h, N_l)."""
        return np.meshgrid(self.h_grid, self.lam_grid, indexing="ij")

    def interpolate(
        self,
        values: np.ndarray,
        h_query: np.ndarray,
        lam_query: np.ndarray,
        failure_penalty: float = 0.0,
    ) -> np.ndarray:
        """Vectorized bilinear interpolation of V(h, lambda) on S, enforcing absorbing failure boundary
        V(h >= h_crit, lambda) = failure_penalty.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.shape != self.shape:
            raise ValueError(f"Expected value array shape {self.shape}, got {v.shape}")

        hq = np.asarray(h_query, dtype=np.float64)
        lq = np.asarray(lam_query, dtype=np.float64)
        out_shape = np.broadcast_shapes(hq.shape, lq.shape)
        hq_b = np.broadcast_to(hq, out_shape)
        lq_b = np.broadcast_to(lq, out_shape)

        failed = hq_b >= self.h_crit
        hc = np.clip(hq_b, self.h_min, self.h_crit)
        lc = np.clip(lq_b, self.lam_min, self.lam_max)

        ih = np.clip(np.searchsorted(self.h_grid, hc, side="right") - 1, 0, len(self.h_grid) - 2)
        il = np.clip(np.searchsorted(self.lam_grid, lc, side="right") - 1, 0, len(self.lam_grid) - 2)

        h0 = self.h_grid[ih]
        h1 = self.h_grid[ih + 1]
        l0 = self.lam_grid[il]
        l1 = self.lam_grid[il + 1]

        wh = (hc - h0) / (h1 - h0)
        wl = (lc - l0) / (l1 - l0)

        v00 = v[ih, il]
        v10 = v[ih + 1, il]
        v01 = v[ih, il + 1]
        v11 = v[ih + 1, il + 1]

        interp_val = (
            (1.0 - wh) * (1.0 - wl) * v00
            + wh * (1.0 - wl) * v10
            + (1.0 - wh) * wl * v01
            + wh * wl * v11
        )
        return np.where(failed, failure_penalty, interp_val)

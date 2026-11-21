from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np
from imdp_phm.core.params import StructuralConstants


class MaintenanceAction(IntEnum):
    CONTINUE = 0
    NDE_INSPECT = 1
    NDE_INSPECTION = 1
    PREVENTIVE_REPLACE = 2


@dataclass(frozen=True)
class MaintenanceCostModel:
    c_oper_base: float = 1.0
    c_oper_damage_quad: float = 8.0
    c_inspection: float = 25.0
    inspection_hazard_relief: float = 0.55
    c_preventive: float = 250.0
    c_failure: float = 3500.0
    discount_factor: float = 0.995

    def stage_cost_grid(self, grid: "StateGrid2D") -> np.ndarray:
        h_mesh, _ = grid.meshgrid()
        failed = h_mesh >= grid.h_crit - 1e-12
        c_cont = np.where(
            failed,
            self.c_failure,
            self.c_oper_base + self.c_oper_damage_quad * ((np.clip(h_mesh / grid.h_crit, 0.0, 1.0)) ** 2),
        )
        c_insp = np.where(failed, self.c_failure, c_cont + self.c_inspection)
        c_prev = np.where(failed, self.c_failure, self.c_preventive)
        return np.stack([c_cont, c_insp, c_prev], axis=-1)

    def step_cost(self, h: float, action: MaintenanceAction, h_crit: float = 2.50) -> float:
        if h >= h_crit - 1e-12:
            return float(self.c_failure)
        if action == MaintenanceAction.PREVENTIVE_REPLACE:
            return float(self.c_preventive)
        c_op = float(self.c_oper_base + self.c_oper_damage_quad * ((np.clip(h / h_crit, 0.0, 1.0)) ** 2))
        return c_op + float(self.c_inspection) if int(action) == int(MaintenanceAction.NDE_INSPECT) else c_op


@dataclass
class StateGrid2D:
    h_grid: np.ndarray
    lam_grid: np.ndarray
    h0: float = 0.10
    h_crit: float = 2.50

    @classmethod
    def uniform(
        cls,
        n_h: int = 41,
        n_lam: int = 25,
        lam_min: float = 0.05,
        lam_max: float = 2.50,
        constants: Optional[StructuralConstants] = None,
    ) -> "StateGrid2D":
        c = constants or StructuralConstants()
        return cls(
            np.linspace(0.0, c.h_crit, n_h, dtype=np.float64),
            np.linspace(lam_min, lam_max, n_lam, dtype=np.float64),
            c.h0,
            c.h_crit,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.h_grid), len(self.lam_grid))

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.h_grid, self.lam_grid, indexing="ij")

    def interpolate_bilinear(self, v_table: np.ndarray, h_query: np.ndarray, lam_query: np.ndarray) -> np.ndarray:
        v = np.asarray(v_table, dtype=np.float64)
        hq = np.clip(h_query, self.h_grid[0], self.h_grid[-1])
        lq = np.clip(lam_query, self.lam_grid[0], self.lam_grid[-1])
        ih = np.clip(np.searchsorted(self.h_grid, hq, side="right") - 1, 0, len(self.h_grid) - 2)
        wh = (hq - self.h_grid[ih]) / (self.h_grid[ih + 1] - self.h_grid[ih])
        il = np.clip(np.searchsorted(self.lam_grid, lq, side="right") - 1, 0, len(self.lam_grid) - 2)
        wl = (lq - self.lam_grid[il]) / (self.lam_grid[il + 1] - self.lam_grid[il])
        return (
            (1.0 - wh) * (1.0 - wl) * v[ih, il]
            + wh * (1.0 - wl) * v[ih + 1, il]
            + (1.0 - wh) * wl * v[ih, il + 1]
            + wh * wl * v[ih + 1, il + 1]
        )

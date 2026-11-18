"""WP3 Competitive Maintenance Policy Baselines & Deep Recurrent RUL Model:
1. IMDPPolicy       : Proposed Minimax Imprecise MDP policy over convex credal set Theta (WP1 + WP2).
2. MLEMDPPolicy     : Classical point-estimate MDP baseline ignoring epistemic uncertainty.
3. ThresholdPolicy  : Industrial heuristic damage/hazard threshold policy.
4. LSTMRULPolicy    : Deep PyTorch LSTM Remaining Useful Life (RUL) predictive maintenance baseline.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from imdp_phm.core.params import StructuralConstants
from imdp_phm.inference.credal import CredalParameterSet
from imdp_phm.inference.mle import (
    MacroEpochDataset,
    MLEEstimationResult,
    fit_hawkes_merton_mle,
)
from imdp_phm.solver.bellman import IMDPBellmanResult, run_wp1_wp2_pipeline
from imdp_phm.solver.grid import MaintenanceAction, MaintenanceCostModel, StateGrid2D
from imdp_phm.solver.transition import PointMDPResult, solve_point_mdp


class MaintenancePolicy(ABC):
    """Unified stateful/memoryless policy interface for closed-loop WP3 evaluation."""

    name: str

    def reset(self) -> None:
        """Reset internal trajectory history at episode start or after preventive replacement."""
        pass

    def observe(self, h: float, lam: float, jump_count: int = 0) -> None:
        """Ingest new macro-epoch telemetry observation (h_k, lam_k, dN_k)."""
        pass

    @abstractmethod
    def select_action(self, h: float, lam: float) -> MaintenanceAction:
        """Select discrete maintenance action a_k in {CONTINUE, NDE_INSPECT, PREVENTIVE_REPLACE}."""
        raise NotImplementedError

    def step(self, h: float, lam: float, jump_count: int = 0) -> MaintenanceAction:
        """Record observation and return maintenance decision in one call."""
        self.observe(h=h, lam=lam, jump_count=jump_count)
        return self.select_action(h=h, lam=lam)


@dataclass
class IMDPPolicy(MaintenancePolicy):
    """Proposed robust minimax policy solving V*(h, lam) over convex credal set Theta."""
    grid: StateGrid2D
    imdp_result: IMDPBellmanResult
    credal_set: Optional[CredalParameterSet] = None
    name: str = "IMDP-PHM"

    def select_action(self, h: float, lam: float) -> MaintenanceAction:
        return self.imdp_result.action_at(self.grid, h=h, lam=lam)

    @classmethod
    def from_dataset(
        cls,
        dataset: MacroEpochDataset,
        grid: Optional[StateGrid2D] = None,
        cost_model: Optional[MaintenanceCostModel] = None,
        constants: Optional[StructuralConstants] = None,
        z_mult: float = 1.96,
        tol: float = 1e-5,
        seed: int = 42,
    ) -> tuple["IMDPPolicy", MLEEstimationResult, CredalParameterSet, IMDPBellmanResult]:
        consts = constants or StructuralConstants()
        g = grid or StateGrid2D.uniform(n_h=41, n_lam=25, lam_min=0.05, lam_max=2.50, constants=consts)
        mle_res, credal, imdp_res = run_wp1_wp2_pipeline(
            dataset=dataset,
            grid=g,
            cost_model=cost_model,
            constants=consts,
            z_mult=z_mult,
            tol=tol,
            seed=seed,
        )
        return cls(grid=g, imdp_result=imdp_res, credal_set=credal), mle_res, credal, imdp_res


@dataclass
class MLEMDPPolicy(MaintenancePolicy):
    """Classical certainty-equivalent MDP baseline using point-estimate theta_hat without credal sets."""
    grid: StateGrid2D
    point_result: PointMDPResult
    mle_result: Optional[MLEEstimationResult] = None
    name: str = "MLE-MDP"

    def select_action(self, h: float, lam: float) -> MaintenanceAction:
        return self.point_result.action_at(self.grid, h=h, lam=lam)

    @classmethod
    def from_dataset(
        cls,
        dataset: MacroEpochDataset,
        grid: Optional[StateGrid2D] = None,
        cost_model: Optional[MaintenanceCostModel] = None,
        constants: Optional[StructuralConstants] = None,
        tol: float = 1e-6,
    ) -> tuple["MLEMDPPolicy", MLEEstimationResult, PointMDPResult]:
        consts = constants or StructuralConstants()
        g = grid or StateGrid2D.uniform(n_h=41, n_lam=25, lam_min=0.05, lam_max=2.50, constants=consts)
        cm = cost_model or MaintenanceCostModel()
        mle_res = fit_hawkes_merton_mle(dataset, constants=consts)
        point_res = solve_point_mdp(
            grid=g,
            params=mle_res.effective_params,
            cost_model=cm,
            constants=consts,
            tol=tol,
        )
        return cls(grid=g, point_result=point_res, mle_result=mle_res), mle_res, point_res


@dataclass
class ThresholdPolicy(MaintenancePolicy):
    """Industrial heuristic threshold policy on cumulative damage H_t and optional hazard alert lambda_th."""
    h_inspect: float = 1.375      # 0.55 * H_crit
    h_replace: float = 2.000      # 0.80 * H_crit
    lam_inspect: Optional[float] = 0.85
    h_crit: float = 2.50
    name: str = "Threshold"

    def __post_init__(self) -> None:
        if not (0.0 < self.h_inspect < self.h_replace <= self.h_crit):
            raise ValueError("Thresholds must satisfy 0 < h_inspect < h_replace <= h_crit.")

    def select_action(self, h: float, lam: float) -> MaintenanceAction:
        if h >= self.h_replace:
            return MaintenanceAction.PREVENTIVE_REPLACE
        if h >= self.h_inspect or (self.lam_inspect is not None and lam >= self.lam_inspect):
            return MaintenanceAction.NDE_INSPECT
        return MaintenanceAction.CONTINUE


class RULLSTMNet(nn.Module):
    """Compact sequence-to-one LSTM network predicting normalized Remaining Useful Life (RUL)."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class LSTMTrainingReport:
    epochs: int
    n_sequences: int
    final_train_loss: float
    train_rmse_hours: float
    loss_history: list[float]


@dataclass
class LSTMRULPolicy(MaintenancePolicy):
    """Deep LSTM Remaining Useful Life (RUL) predictive maintenance baseline (Concept Note WP3).

    Ingests sliding windows of observable macro-epoch telemetry [h / h_crit, dh / (0.1*h_crit),
    shock_count, ewma_shock_rate], predicts RUL in hours, and triggers NDE inspection or preventive
    replacement when predicted RUL drops below operational lead thresholds.
    """
    constants: StructuralConstants = field(default_factory=StructuralConstants)
    seq_len: int = 16
    hidden_dim: int = 32
    max_rul_hours: float = 100.0
    rul_inspect_hours: float = 24.0
    rul_replace_hours: float = 12.0
    hard_replace_ratio: float = 0.96
    seed: int = 42
    name: str = "LSTM-RUL"

    def __post_init__(self) -> None:
        if not (0.0 < self.rul_replace_hours < self.rul_inspect_hours < self.max_rul_hours):
            raise ValueError("RUL thresholds must satisfy 0 < rul_replace_hours < rul_inspect_hours < max_rul_hours.")
        torch.manual_seed(self.seed)
        self.model = RULLSTMNet(input_dim=4, hidden_dim=self.hidden_dim, num_layers=1)
        self.model.eval()
        self.training_report: Optional[LSTMTrainingReport] = None
        self._history: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._prev_h: float = self.constants.h0
        self._ewma_rate: float = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._prev_h = self.constants.h0
        self._ewma_rate = 0.0

    def _featurize_step(self, h: float, dh: float, jump_count: int, ewma_rate: float) -> np.ndarray:
        h_crit = self.constants.h_crit
        return np.array(
            [
                np.clip(h / h_crit, 0.0, 1.5),
                np.clip(dh / (0.10 * h_crit), -1.0, 5.0),
                float(jump_count),
                np.clip(ewma_rate / 2.5, 0.0, 3.0),
            ],
            dtype=np.float32,
        )

    def observe(self, h: float, lam: float, jump_count: int = 0) -> None:
        dh = max(0.0, h - self._prev_h) if len(self._history) > 0 else 0.0
        inst_rate = float(jump_count) / max(self.constants.dt, 1e-6)
        self._ewma_rate = 0.85 * self._ewma_rate + 0.15 * inst_rate
        feat = self._featurize_step(h=h, dh=dh, jump_count=jump_count, ewma_rate=self._ewma_rate)
        self._history.append(feat)
        self._prev_h = float(h)

    @torch.no_grad()
    def predict_rul_hours(self) -> float:
        if len(self._history) == 0:
            return self.max_rul_hours
        seq = list(self._history)
        if len(seq) < self.seq_len:
            pad = [seq[0]] * (self.seq_len - len(seq))
            seq = pad + seq
        x_tensor = torch.from_numpy(np.asarray(seq, dtype=np.float32)[None, :, :])
        self.model.eval()
        pred_norm = float(self.model(x_tensor).item())
        return float(np.clip(pred_norm * self.max_rul_hours, 0.0, self.max_rul_hours))

    def select_action(self, h: float, lam: float) -> MaintenanceAction:
        if h >= self.hard_replace_ratio * self.constants.h_crit:
            return MaintenanceAction.PREVENTIVE_REPLACE
        # Avoid padded cold-start false alarms during initial warm-up when damage is still nominal
        if len(self._history) < max(2, self.seq_len // 2) and h < 0.45 * self.constants.h_crit:
            return MaintenanceAction.CONTINUE
        rul_hat = self.predict_rul_hours()
        if rul_hat <= self.rul_replace_hours:
            return MaintenanceAction.PREVENTIVE_REPLACE
        if rul_hat <= self.rul_inspect_hours:
            return MaintenanceAction.NDE_INSPECT
        return MaintenanceAction.CONTINUE

    def fit_from_dataset(
        self,
        dataset: MacroEpochDataset,
        epochs: int = 35,
        lr: float = 3e-3,
        batch_size: int = 128,
        stride: int = 2,
    ) -> LSTMTrainingReport:
        """Train LSTM RUL regressor on run-to-failure macro-epoch trajectories."""
        torch.manual_seed(self.seed)
        h_paths = dataset.h_paths          # (M, K+1)
        jumps = dataset.jump_indicators    # (M, K)
        dt = dataset.dt
        h_crit = self.constants.h_crit
        m_paths, k_steps = jumps.shape

        x_windows = []
        y_targets = []

        for m in range(m_paths):
            fail_idxs = np.flatnonzero(h_paths[m, 1:] >= h_crit)
            end_k = int(fail_idxs[0]) + 1 if len(fail_idxs) > 0 else k_steps
            if end_k < self.seq_len + 2:
                continue

            feats = np.empty((end_k, 4), dtype=np.float32)
            ewma = 0.0
            for k in range(end_k):
                h_curr = float(h_paths[m, k + 1])
                dh = max(0.0, float(h_paths[m, k + 1] - h_paths[m, k]))
                dn = int(jumps[m, k])
                ewma = 0.85 * ewma + 0.15 * (dn / max(dt, 1e-6))
                feats[k] = self._featurize_step(h=h_curr, dh=dh, jump_count=dn, ewma_rate=ewma)

            for t_end in range(self.seq_len, end_k, max(1, stride)):
                rul_hours = min((end_k - t_end) * dt, self.max_rul_hours)
                x_windows.append(feats[t_end - self.seq_len : t_end])
                y_targets.append(rul_hours / self.max_rul_hours)

        if len(x_windows) == 0:
            rep = LSTMTrainingReport(epochs=0, n_sequences=0, final_train_loss=0.0, train_rmse_hours=0.0, loss_history=[])
            self.training_report = rep
            return rep

        x_arr = torch.from_numpy(np.asarray(x_windows, dtype=np.float32))
        y_arr = torch.from_numpy(np.asarray(y_targets, dtype=np.float32))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss(beta=0.1)
        rng = np.random.Generator(np.random.PCG64(self.seed))
        n_seq = len(x_arr)
        loss_hist: list[float] = []

        self.model.train()
        for _ in range(epochs):
            perm = rng.permutation(n_seq)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_seq, batch_size):
                idx = torch.from_numpy(perm[start : start + batch_size])
                optimizer.zero_grad()
                pred = self.model(x_arr[idx])
                loss = criterion(pred, y_arr[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            loss_hist.append(epoch_loss / max(n_batches, 1))

        self.model.eval()
        with torch.no_grad():
            preds_all = self.model(x_arr).numpy() * self.max_rul_hours
            trues_all = y_arr.numpy() * self.max_rul_hours
            rmse_hours = float(np.sqrt(np.mean((preds_all - trues_all) ** 2)))

        rep = LSTMTrainingReport(
            epochs=epochs,
            n_sequences=n_seq,
            final_train_loss=loss_hist[-1],
            train_rmse_hours=rmse_hours,
            loss_history=loss_hist,
        )
        self.training_report = rep
        return rep

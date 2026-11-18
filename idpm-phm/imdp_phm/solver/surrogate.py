"""Gaussian Process (GP) Surrogate-Assisted Upper Transition Expectation Operator
T_bar_{Delta t}^a V(h, lambda) = sup_{theta in Theta} E_theta[V(H_{t+Delta t}, lambda_{t+Delta t}) | h, lam, a]
resolving interior extrema over convex credal sets Theta subset R^8 (Concept Note Sec. 2 & Objective 3).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from imdp_phm.core.params import HawkesMertonParams
from imdp_phm.inference.credal import CredalParameterSet
from imdp_phm.solver.grid import MaintenanceAction
from imdp_phm.solver.transition import SingleStepQuadratureOperator


@dataclass
class UpperExpectationResult:
    upper_expectation: np.ndarray
    worst_case_theta: np.ndarray
    gp_r2_score: float
    method: str


class GaussianProcessUpperOperator:
    def __init__(
        self,
        quad_operator: SingleStepQuadratureOperator,
        credal_set: CredalParameterSet,
        n_design: int = 36,
        n_restarts: int = 4,
        nugget: float = 1e-7,
        seed: int = 42,
    ) -> None:
        self.op = quad_operator
        self.credal_set = credal_set
        self.n_design = int(n_design)
        self.n_restarts = int(n_restarts)
        self.nugget = float(nugget)
        self.seed = int(seed)

        self._design_theta = self._build_design_points()
        self._active_dims = np.where(self.credal_set.upper - self.credal_set.lower > 1e-12)[0]
        self._design_scaled = self._scale_theta(self._design_theta)

        self._length_scales = np.full(max(len(self._active_dims), 1), 3.50, dtype=np.float64)
        self._k_mat = self._matern52_matrix(self._design_scaled, self._design_scaled)
        self._k_mat[np.diag_indices_from(self._k_mat)] += max(self.nugget, 1e-6)
        self._cho = cho_factor(self._k_mat, lower=True)

    def _build_design_points(self) -> np.ndarray:
        lhs = self.credal_set.sample_latin_hypercube(n_samples=self.n_design, seed=self.seed)
        center = self.credal_set.project(0.5 * (self.credal_set.lower + self.credal_set.upper))[None, :]
        worst_anchor = self.credal_set.project(
            np.array([
                self.credal_set.upper[0],
                self.credal_set.upper[1],
                self.credal_set.upper[2],
                self.credal_set.upper[3],
                self.credal_set.lower[4],
                self.credal_set.upper[5],
                self.credal_set.upper[6],
                self.credal_set.upper[7],
            ], dtype=np.float64)
        )[None, :]
        best_anchor = self.credal_set.project(
            np.array([
                self.credal_set.lower[0],
                self.credal_set.lower[1],
                self.credal_set.lower[2],
                self.credal_set.lower[3],
                self.credal_set.upper[4],
                self.credal_set.lower[5],
                self.credal_set.lower[6],
                self.credal_set.lower[7],
            ], dtype=np.float64)
        )[None, :]
        pts = np.vstack([center, worst_anchor, best_anchor, lhs])
        return np.unique(np.round(pts, decimals=12), axis=0)

    def _scale_theta(self, theta_batch: np.ndarray) -> np.ndarray:
        if len(self._active_dims) == 0:
            return np.zeros((len(theta_batch), 1), dtype=np.float64)
        lo = self.credal_set.lower[self._active_dims]
        hi = self.credal_set.upper[self._active_dims]
        return 2.0 * (theta_batch[:, self._active_dims] - lo) / (hi - lo) - 1.0

    def _unscale_theta(self, x_scaled: np.ndarray) -> np.ndarray:
        v = 0.5 * (self.credal_set.lower + self.credal_set.upper)
        if len(self._active_dims) > 0:
            lo = self.credal_set.lower[self._active_dims]
            hi = self.credal_set.upper[self._active_dims]
            v[self._active_dims] = lo + 0.5 * (np.clip(x_scaled, -1.0, 1.0) + 1.0) * (hi - lo)
        return self.credal_set.project(v)

    def _matern52_matrix(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        diff = (x1[:, None, :] - x2[None, :, :]) / self._length_scales[None, None, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-16)
        s5_r = np.sqrt(5.0) * r
        return (1.0 + s5_r + (5.0 / 3.0) * (r ** 2)) * np.exp(-s5_r)

    def _matern52_vec_and_grad(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = (x[None, :] - self._design_scaled) / self._length_scales[None, :]
        r_sq = np.sum(diff ** 2, axis=-1)
        r = np.sqrt(r_sq + 1e-16)
        s5_r = np.sqrt(5.0) * r
        exp_term = np.exp(-s5_r)
        k_vec = (1.0 + s5_r + (5.0 / 3.0) * r_sq) * exp_term
        dk_dr_over_r = -(5.0 / 3.0) * (1.0 + s5_r) * exp_term
        dk_dx = dk_dr_over_r[:, None] * (diff / self._length_scales[None, :])
        return k_vec, dk_dx

    def _evaluate_exact_batch(self, v_table: np.ndarray, theta_batch: np.ndarray) -> np.ndarray:
        m = len(theta_batch)
        n_h, n_lam = self.op.grid.shape
        out = np.empty((m, n_h, n_lam, 3), dtype=np.float64)
        for idx in range(m):
            p = HawkesMertonParams.from_vector(theta_batch[idx])
            out[idx] = self.op.all_action_expectations(v_table, p)
        return out

    def upper_transition_surrogate(
        self,
        v_table: np.ndarray,
        verify_exact_candidates: bool = True,
        extra_probe_states: Optional[list[tuple[int, int, int]]] = None,
        max_verify_candidates: int = 32,
        precomputed_design_vals: Optional[np.ndarray] = None,
        extra_eval_thetas: Optional[np.ndarray] = None,
        extra_eval_vals: Optional[np.ndarray] = None,
        batch_eval_fn: Optional[object] = None,
    ) -> UpperExpectationResult:
        n_h, n_lam = self.op.grid.shape
        if len(self._active_dims) == 0:
            theta0 = self._design_theta[0]
            ev0 = self.op.all_action_expectations(v_table, HawkesMertonParams.from_vector(theta0))
            w_theta = np.broadcast_to(theta0[None, None, None, :], (n_h, n_lam, 3, 8)).copy()
            return UpperExpectationResult(
                upper_expectation=ev0,
                worst_case_theta=w_theta,
                gp_r2_score=1.0,
                method="gp_surrogate",
            )

        if precomputed_design_vals is not None:
            y_train_4d = precomputed_design_vals
        else:
            y_train_4d = self._evaluate_exact_batch(v_table, self._design_theta)
        m_pts = y_train_4d.shape[0]
        y_flat = y_train_4d.reshape(m_pts, -1)

        y_mean = np.mean(y_flat, axis=0)
        y_std = np.std(y_flat, axis=0)
        active_outputs = y_std > 1e-10
        y_scale = np.where(active_outputs, y_std, 1.0)
        y_norm = (y_flat - y_mean) / y_scale

        alpha_weights = cho_solve(self._cho, y_norm)

        if np.any(active_outputs):
            k_inv_diag = np.diag(cho_solve(self._cho, np.eye(m_pts)))
            loo_err = alpha_weights[:, active_outputs] / k_inv_diag[:, None]
            ss_res = float(np.sum(loo_err ** 2))
            ss_tot = float(np.sum(y_norm[:, active_outputs] ** 2))
            r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
        else:
            r2 = 1.0

        verts = self.credal_set.vertices(project_subcritical=True)
        if len(verts) > 128:
            rng_v = np.random.Generator(np.random.PCG64(self.seed + 55))
            verts = verts[rng_v.choice(len(verts), size=128, replace=False)]
        lhs_cand = self.credal_set.sample_latin_hypercube(n_samples=128, seed=self.seed + 101)
        cand_theta = np.vstack([self._design_theta, verts, lhs_cand])
        cand_scaled = self._scale_theta(cand_theta)
        k_cand = self._matern52_matrix(cand_scaled, self._design_scaled)
        pred_norm = k_cand @ alpha_weights
        pred_flat = pred_norm * y_scale + y_mean

        best_idx_per_out = np.argmax(pred_flat, axis=0)
        best_val_per_out = pred_flat[best_idx_per_out, np.arange(pred_flat.shape[1])]
        best_theta_per_out = cand_theta[best_idx_per_out]

        d_act = len(self._active_dims)
        bounds_scaled = [(-1.0, 1.0)] * d_act
        refined_thetas = []

        var_order = np.argsort(-y_std)
        probe_indices = [int(idx) for idx in var_order[:4] if active_outputs[idx]]
        for h_frac in (0.25, 0.45, 0.50, 0.75):
            ih_s = min(int(round(h_frac * (n_h - 1))), n_h - 1)
            for l_frac in (0.35, 0.50, 0.70):
                il_s = min(int(round(l_frac * (n_lam - 1))), n_lam - 1)
                probe_indices.append((ih_s * n_lam + il_s) * 3)
        probe_indices.append(2)
        if extra_probe_states:
            for ih_e, il_e, act_e in extra_probe_states:
                probe_indices.append((int(ih_e) * n_lam + int(il_e)) * 3 + int(act_e))
        probe_states = [
            (idx // (n_lam * 3), (idx // 3) % n_lam, idx % 3)
            for idx in dict.fromkeys(probe_indices)
        ]
        for ih, il, act in probe_states:
            p_idx = (ih * n_lam + il) * 3 + act
            if not active_outputs[p_idx]:
                continue
            w_p = alpha_weights[:, p_idx]

            def neg_gp_and_grad(x_sc: np.ndarray) -> tuple[float, np.ndarray]:
                kv, dkv = self._matern52_vec_and_grad(x_sc)
                val = float(np.dot(kv, w_p))
                grad = dkv.T @ w_p
                return -val, -grad

            x_start = cand_scaled[best_idx_per_out[p_idx]]
            opt = minimize(neg_gp_and_grad, x0=x_start, jac=True, method="L-BFGS-B", bounds=bounds_scaled)
            theta_ref = self._unscale_theta(opt.x)
            refined_thetas.append(theta_ref)

            x_ref_sc = self._scale_theta(theta_ref[None, :])
            pred_ref = (self._matern52_matrix(x_ref_sc, self._design_scaled) @ alpha_weights)[0] * y_scale + y_mean
            improve = pred_ref > best_val_per_out
            best_val_per_out = np.where(improve, pred_ref, best_val_per_out)
            best_theta_per_out[improve] = theta_ref

        if verify_exact_candidates:
            cand_verify = [self._design_theta[1], self._design_theta[2]]
            design_max_per_out = np.max(y_flat, axis=0)
            predicted_gain = np.maximum(0.0, best_val_per_out - design_max_per_out)
            rounded_best = np.round(best_theta_per_out, decimals=8)
            uniq_gp, inv_idx, counts = np.unique(rounded_best, axis=0, return_inverse=True, return_counts=True)
            max_gain_per_uniq = np.zeros(len(uniq_gp), dtype=np.float64)
            np.maximum.at(max_gain_per_uniq, inv_idx, predicted_gain)
            rank_score = max_gain_per_uniq * 100.0 + counts.astype(np.float64)
            top_k = np.argsort(-rank_score)[:max_verify_candidates]
            cand_verify.extend(list(uniq_gp[top_k]))
            cand_verify.extend(refined_thetas)
            unique_thetas = np.array(
                [self.credal_set.project(row) for row in np.unique(np.round(np.asarray(cand_verify, dtype=np.float64), decimals=8), axis=0)],
                dtype=np.float64,
            )

            if batch_eval_fn is not None:
                exact_vals = batch_eval_fn(v_table, unique_thetas).reshape(len(unique_thetas), -1)
            else:
                exact_vals = self._evaluate_exact_batch(v_table, unique_thetas).reshape(len(unique_thetas), -1)
            all_eval_thetas = np.vstack([self._design_theta, unique_thetas])
            all_eval_vals = np.vstack([y_flat, exact_vals])
            if extra_eval_thetas is not None and extra_eval_vals is not None and len(extra_eval_thetas) > 0:
                all_eval_thetas = np.vstack([all_eval_thetas, extra_eval_thetas])
                all_eval_vals = np.vstack([all_eval_vals, extra_eval_vals.reshape(len(extra_eval_thetas), -1)])

            exact_best_idx = np.argmax(all_eval_vals, axis=0)
            upper_flat = all_eval_vals[exact_best_idx, np.arange(all_eval_vals.shape[1])]
            worst_theta_flat = all_eval_thetas[exact_best_idx]
        else:
            upper_flat = best_val_per_out
            worst_theta_flat = best_theta_per_out

        return UpperExpectationResult(
            upper_expectation=upper_flat.reshape(n_h, n_lam, 3),
            worst_case_theta=worst_theta_flat.reshape(n_h, n_lam, 3, 8),
            gp_r2_score=r2,
            method="gp_surrogate",
        )

    def upper_transition_direct(
        self,
        v_table: np.ndarray,
        n_random_interior: int = 48,
    ) -> UpperExpectationResult:
        n_h, n_lam = self.op.grid.shape
        verts = self.credal_set.vertices(project_subcritical=True)
        if len(verts) > 64:
            rng = np.random.Generator(np.random.PCG64(self.seed + 7))
            verts = verts[rng.choice(len(verts), size=64, replace=False)]
        interior = self.credal_set.sample_latin_hypercube(n_samples=n_random_interior, seed=self.seed + 19)
        pool = np.unique(np.round(np.vstack([self._design_theta[:3], verts, interior]), decimals=10), axis=0)

        vals = self._evaluate_exact_batch(v_table, pool).reshape(len(pool), -1)
        best_idx = np.argmax(vals, axis=0)
        upper_flat = vals[best_idx, np.arange(vals.shape[1])]
        worst_theta_flat = pool[best_idx]

        return UpperExpectationResult(
            upper_expectation=upper_flat.reshape(n_h, n_lam, 3),
            worst_case_theta=worst_theta_flat.reshape(n_h, n_lam, 3, 8),
            gp_r2_score=1.0,
            method="exact_direct",
        )

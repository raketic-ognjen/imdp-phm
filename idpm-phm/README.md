# IMDP-PHM: Condition-Based Maintenance under Imprecise Self-Exciting Jump-Diffusion Degradation

Full-stack scientific framework and interactive workbench for **WP1–WP3** (*DFG Sachbeihilfe*, Chair of Reliability Engineering, TU Dortmund University).

## Overview

This repository implements an end-to-end Prognostics and Health Management (PHM) and Condition-Based Maintenance (CBM) framework under epistemic parameter uncertainty:

- **WP1 — Multi-Scale Hypothesis Testing & Credal Identification (`core/`, `multiscale/`, `inference/`):** Separates sub-second kinematic bearing fault signatures (BPFO) from macro-scale degradation epochs ($\Delta t$), models cumulative damage and self-exciting shock intensity $(H_t, \lambda_t)$ via a coupled Hawkes-Merton jump-diffusion SDE, tests $H_0: \alpha_0 = 0$ (NHPP baseline) against $H_1: \alpha_0 > 0$ (NH-Hawkes) using boundary-corrected LRT and $\Delta\text{BIC}$, and constructs convex credal parameter sets $\Theta \subset \mathbb{R}_+^8$ alongside parametric jump-severity probability boxes $[\underline{F}_J, \overline{F}_J]$.
- **WP2 — Surrogate-Assisted Minimax IMDP Solver (`solver/`):** Discretizes the bivariate state space $(h, \lambda)$, compiles row-stochastic sparse transition matrices $P_\theta^a$, and solves the worst-case minimax Bellman optimality equation $V^*(h, \lambda)$ over $\Theta$ via a multi-output Matérn-5/2 Gaussian Process emulator, L-BFGS-B interior optimization, and Howard Policy Iteration.
- **WP3 — Closed-Loop Stress Benchmarking (`benchmark/`):** Evaluates **IMDP-PHM** against certainty-equivalent **MLE-MDP**, deep recurrent **LSTM-RUL**, and heuristic **Threshold** policies using paired Common Random Numbers Monte Carlo rollouts under nominal and load-shift regimes ($\text{CVaR}_{95\%}$ tail risk, uncontained failure rate, and decision lead margin $\Delta\tau_{\text{LIH}}$).
- **Production API & Workbench (`api/`, `frontend/`):** Exposes all WP1–WP3 engines through a typed FastAPI + Pydantic v2 REST service connected to a modular React + Vite visualization interface.

## Repository Structure

```text
imdp-phm/
├── imdp_phm/
│   ├── core/          # Coupled Hawkes-Merton SDE simulator & infinitesimal generator
│   ├── multiscale/    # Fast-to-slow BPFO signal separation & macro-epoch aggregation
│   ├── inference/     # Decoupled MLE, H0 vs H1 LRT/BIC test, convex credal sets & p-boxes
│   ├── solver/        # 2D state grid, GP-assisted upper operator & sparse minimax Bellman solver
│   ├── benchmark/     # IMDP-PHM, MLE-MDP, LSTM-RUL, Threshold policies & CVaR95 evaluator
│   └── api/           # FastAPI REST backend & Pydantic v2 schemas
├── tests/             # Complete pytest suite across WP1, WP2, WP3, and FastAPI routes
├── frontend/          # Modular React + Vite interactive scientific workbench
├── docs/              # DFG Sachbeihilfe project concept note
├── ui_mockups/        # Reference UI architecture previews
├── requirements.txt
└── README.md

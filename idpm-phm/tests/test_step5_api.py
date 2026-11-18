"""End-to-end API contract, Pydantic v2 validation, and JSON payload tests for Step 5 (FastAPI Backend)."""

from imdp_phm.api import LocalAPITestClient


def test_api_health_and_validation_errors():
    """Verify GET /api/v1/health and strict Pydantic 422 validation on invalid physical parameters."""
    client = LocalAPITestClient()
    r_ok = client.get("/api/v1/health")
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert body["status"] == "ok"
    assert body["service"] == "imdp-phm-api"

    r_bad = client.post("/api/v1/wp1/simulate-and-identify", {"true_params": {"a0": -0.01}})
    assert r_bad.status_code == 422


def test_api_wp1_wp2_and_full_pipeline_endpoints():
    """Verify POST endpoints for WP1, WP2, and full end-to-end WP1->WP2->WP3 JSON payloads."""
    client = LocalAPITestClient()

    r_wp1 = client.post(
        "/api/v1/wp1/simulate-and-identify",
        {"n_trajectories": 12, "t_max_hours": 140.0, "seed": 42},
    )
    assert r_wp1.status_code == 200
    d1 = r_wp1.json()
    assert d1["hypothesis_test"]["active_regime"] == "NH-Hawkes"
    assert len(d1["parameter_intervals"]) == 8
    assert len(d1["pbox"]["y_grid"]) == 60
    assert len(d1["sample_trajectories"]) == 3

    payload = {
        "wp2_config": {
            "wp1_config": {"n_trajectories": 12, "t_max_hours": 140.0, "seed": 42},
            "grid": {"n_h": 25, "n_lam": 15, "lam_min": 0.05, "lam_max": 2.50},
            "tol": 1e-4,
        },
        "n_eval_episodes": 60,
        "horizon_hours": 130.0,
        "lstm_epochs": 15,
    }
    r_full = client.post("/api/v1/pipeline/full", payload)
    assert r_full.status_code == 200
    df = r_full.json()

    assert set(df.keys()) == {"wp1", "wp2", "wp3", "total_pipeline_time_ms"}
    assert df["wp2"]["converged"] is True
    assert len(df["wp2"]["imdp_policy_map"]) == 25
    assert len(df["wp2"]["imdp_policy_map"][0]) == 15
    assert df["wp2"]["v_imdp_pristine"] >= df["wp2"]["v_mle_pristine"]

    assert len(df["wp3"]["nominal_benchmark"]) == 4
    assert len(df["wp3"]["load_shift_benchmark"]) == 4
    for row in df["wp3"]["load_shift_benchmark"]:
        assert row["cvar_95"] >= row["mean_cost"]
        assert 0.0 <= row["failure_rate"] <= 1.0

"""API tests for /api/empirical/catalog and /api/empirical/run endpoints."""

from pathlib import Path
import numpy as np

from imdp_phm.api.app import LocalAPITestClient


def test_empirical_api_catalog_and_run_from_cache(tmp_path: Path):
    rng = np.random.default_rng(2026)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # Create cached trajectory for IMS_2nd_test_B1 with clear self-exciting shock cluster at end
    n = 120
    t_hrs = np.arange(1, n + 1, dtype=np.float64) * (10.0 / 60.0)
    v_max = 0.12 + 0.015 * rng.standard_normal(n)
    v_max[95:] += np.linspace(0.15, 0.95, 25)
    np.savez_compressed(cache_dir / "IMS_2nd_test_B1.npz", v_max=v_max, epoch_times=t_hrs)

    client = LocalAPITestClient()
    cat_resp = client.get("/api/empirical/catalog")
    assert cat_resp.status_code == 200
    assert len(cat_resp.json()["items"]) >= 6

    run_resp = client.post(
        "/api/empirical/run",
        {
            "target_id": "IMS_2nd_test_B1",
            "z_thresh": 4.0,
            "run_wp2": True,
            "run_wp3": False,
            "cache_dir": str(cache_dir),
        },
    )
    assert run_resp.status_code == 200
    body = run_resp.json()
    assert body["target_id"] == "IMS_2nd_test_B1"
    assert len(body["trajectories"]) == 1
    assert body["diagnostics"]["n_trajectories"] == 1
    assert "ks_pvalue_h1" in body["diagnostics"]
    assert "lb_pvalue_5_h1" in body["diagnostics"]
    assert body["solver"] is not None
    assert body["solver"]["converged"] is True

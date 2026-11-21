"""End-to-end HTTP & Pydantic schema verification tests for Step 5 (FastAPI Backend)."""

import json
import logging
import threading
import traceback
import urllib.error
import urllib.request

from imdp_phm.api import create_fallback_server

# Привремено: пустимо логове сервера да се виде у терминалу
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _http_get(base_url: str, path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            # Ако 500 није JSON — вратимо сиров текст, да не изгубимо поруку
            return e.code, {"_raw_response": raw}


def test_api_health_and_validation_errors():
    server = create_fallback_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        st, health = _http_get(base_url, "/api/health")
        assert st == 200 and health["status"] == "ok"
        assert "core" in health["modules_ready"]

        st_bad, err_body = _http_post(base_url, "/api/wp1/simulate", {"params": {"a0": -1.0}})
        assert st_bad == 422 and "detail" in err_body
    finally:
        server.shutdown()
        server.server_close()


def test_api_wp1_wp2_and_full_pipeline_endpoints():
    server = create_fallback_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        st, sim_out = _http_post(base_url, "/api/wp1/simulate", {"n_paths": 8, "t_max": 120.0, "max_preview_paths": 3, "seed": 42})
        assert st == 200
        assert len(sim_out["trajectories"]) == 3

        st, solv_out = _http_post(base_url, "/api/wp2/solve", {"n_h": 25, "n_lam": 15, "tol": 1e-4, "seed": 42})
        assert st == 200
        assert solv_out["converged"] is True
        assert solv_out["v_imdp_pristine"] >= solv_out["v_mle_pristine"]

        payload = {
            "true_params": {"a0": 2.8e-3, "sigma0": 2.0e-3, "mu0": 0.30, "alpha0": 0.40, "beta": 0.90, "gamma": 1.30, "mu_J": -2.3, "sigma_J": 0.58},
            "n_train_paths": 6,
            "n_eval_episodes": 25,
            "horizon_hours": 120.0,
            "seed": 42,
        }
        st, full_out = _http_post(base_url, "/api/pipeline/run", payload)

        print("\n" + "=" * 70)
        print(f"PIPELINE STATUS: {st}")
        print("PIPELINE BODY:")
        print(json.dumps(full_out, indent=2, default=str)[:5000])
        print("=" * 70 + "\n")

        # Ако је 500, покушајмо директан позив функције (да одвојимо HTTP од pipeline-а)
        if st != 200:
            print(">>> 500 detected, probing direct call of the pipeline function...")
            try:
                # Име функције у api.py — прилагодите ако се зове другачије
                from imdp_phm.api import run_full_pipeline  # noqa: F401
                direct = run_full_pipeline(**payload)
                print("DIRECT CALL OK; type:", type(direct))
                try:
                    json.dumps(direct, default=None, allow_nan=False)
                    print("DIRECT RESULT is JSON-serializable")
                except (TypeError, ValueError) as te:
                    print("DIRECT RESULT is NOT JSON-serializable:", te)
            except ImportError as ie:
                print("Cannot probe direct call (import name may differ):", ie)
            except Exception:
                print("DIRECT CALL raised:")
                traceback.print_exc()

        assert st == 200
        assert len(full_out["benchmark"]["nominal_results"]) == 4
        assert full_out["total_runtime_sec"] > 0.0
    finally:
        server.shutdown()
        server.server_close()

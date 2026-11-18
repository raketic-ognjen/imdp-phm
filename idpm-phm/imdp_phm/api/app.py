"""FastAPI Production Backend & Built-In HTTP Server for IMDP-PHM Platform.

Run with Uvicorn (when fastapi & uvicorn are installed):
    uvicorn imdp_phm.api.app:app --host 0.0.0.0 --port 8000 --reload

Or run directly with Python standard library HTTP server fallback:
    python -m imdp_phm.api.app
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Callable
from pydantic import ValidationError

from imdp_phm.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    DefaultsResponse,
    FullStudyRequest,
    FullStudyResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    SimulationRequest,
    SimulationResponse,
    SolverRequest,
    SolverResponse,
)
from imdp_phm.api.service import (
    get_defaults,
    get_health,
    run_benchmark,
    run_full_study,
    run_inference,
    run_simulation,
    run_solver,
)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="IMDP-PHM Scientific API",
        description=(
            "Condition-Based Maintenance under Imprecise Self-Exciting Jump-Diffusion Degradation "
            "(WP1 Hawkes-Merton Inference, WP2 Minimax Bellman Solver, WP3 Competitive Benchmark)."
        ),
        version="0.4.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    def api_health() -> HealthResponse:
        return get_health()

    @app.get("/api/defaults", response_model=DefaultsResponse, tags=["System"])
    def api_defaults() -> DefaultsResponse:
        return get_defaults()

    @app.post("/api/wp1/simulate", response_model=SimulationResponse, tags=["WP1: Stochastic Degradation & Inference"])
    def api_wp1_simulate(req: SimulationRequest) -> SimulationResponse:
        return run_simulation(req)

    @app.post("/api/wp1/infer", response_model=InferenceResponse, tags=["WP1: Stochastic Degradation & Inference"])
    def api_wp1_infer(req: InferenceRequest) -> InferenceResponse:
        return run_inference(req)

    @app.post("/api/wp2/solve", response_model=SolverResponse, tags=["WP2: Minimax IMDP Solver"])
    def api_wp2_solve(req: SolverRequest) -> SolverResponse:
        return run_solver(req)

    @app.post("/api/wp3/benchmark", response_model=BenchmarkResponse, tags=["WP3: Competitive Benchmark"])
    def api_wp3_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
        return run_benchmark(req)

    @app.post("/api/pipeline/run", response_model=FullStudyResponse, tags=["Full Study Orchestrator"])
    def api_pipeline_run(req: FullStudyRequest) -> FullStudyResponse:
        return run_full_study(req)

except ImportError:
    app = None


_POST_ROUTES: dict[str, tuple[type, Callable[[Any], Any]]] = {
    "/api/wp1/simulate": (SimulationRequest, run_simulation),
    "/api/wp1/infer": (InferenceRequest, run_inference),
    "/api/wp2/solve": (SolverRequest, run_solver),
    "/api/wp3/benchmark": (BenchmarkRequest, run_benchmark),
    "/api/pipeline/run": (FullStudyRequest, run_full_study),
}


class IMDPRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP/JSON handler mirroring the FastAPI routes with CORS support."""

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"status": "ok"})

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/health":
            self._send_json(200, get_health().model_dump())
        elif path == "/api/defaults":
            self._send_json(200, get_defaults().model_dump())
        else:
            self._send_json(404, {"detail": f"Not found: {path}"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path not in _POST_ROUTES:
            self._send_json(404, {"detail": f"Not found: {path}"})
            return
        req_cls, handler_fn = _POST_ROUTES[path]
        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
            raw_data = json.loads(raw_bytes.decode("utf-8"))
            req_obj = req_cls.model_validate(raw_data)
            resp_obj = handler_fn(req_obj)
            self._send_json(200, resp_obj.model_dump())
        except ValidationError as exc:
            self._send_json(422, {"detail": exc.errors()})
        except Exception as exc:
            self._send_json(500, {"detail": str(exc)})


def create_fallback_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), IMDPRequestHandler)


if __name__ == "__main__":
    srv = create_fallback_server("0.0.0.0", 8000)
    print("Serving IMDP-PHM API on http://0.0.0.0:8000")
    srv.serve_forever()

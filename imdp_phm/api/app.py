"""FastAPI Production Backend & Built-In HTTP Server for IMDP-PHM Platform."""

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse
from pydantic import BaseModel, ValidationError

from imdp_phm import __version__
from imdp_phm.api.schemas import (
    EmpiricalCatalogResponse,
    EmpiricalRunRequest,
    EmpiricalRunResponse,
    FullStudyResponse,
    HealthResponse,
    DefaultsResponse,
    SimulationRequest,
    SimulationResponse,
    InferenceRequest,
    InferenceResponse,
    SolverRequest,
    SolverResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    FullStudyRequest,
)
from imdp_phm.api.service import (
    get_defaults,
    get_empirical_catalog,
    get_health,
    run_benchmark,
    run_empirical_study,
    run_full_study,
    run_inference,
    run_simulation,
    run_solver,
)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    FastAPI = None
    CORSMiddleware = None
    HAS_FASTAPI = False


@dataclass
class LocalResponse:
    status_code: int
    body: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.body


class LocalAPITestClient:
    """Zero-dependency HTTP-like test client executing identical Pydantic validation and route handlers."""
    def __init__(self) -> None:
        self._get_routes: dict[str, Callable[[], BaseModel]] = {
            "/api/health": get_health,
            "/api/defaults": get_defaults,
            "/api/empirical/catalog": get_empirical_catalog,
        }
        self._post_routes: dict[str, tuple[type[BaseModel], Callable[[Any], BaseModel]]] = {
            "/api/wp1/simulate": (SimulationRequest, run_simulation),
            "/api/wp1/infer": (InferenceRequest, run_inference),
            "/api/wp2/solve": (SolverRequest, run_solver),
            "/api/wp3/benchmark": (BenchmarkRequest, run_benchmark),
            "/api/pipeline/run": (FullStudyRequest, run_full_study),
            "/api/empirical/run": (EmpiricalRunRequest, run_empirical_study),
        }

    def get(self, path: str) -> LocalResponse:
        clean_path = path.split("?")[0]
        fn = self._get_routes.get(clean_path)
        if fn is None:
            return LocalResponse(404, {"detail": f"Not Found: {path}"})
        res = fn()
        return LocalResponse(200, json.loads(res.model_dump_json()))

    def post(self, path: str, json_body: Optional[dict[str, Any]] = None) -> LocalResponse:
        entry = self._post_routes.get(path)
        if entry is None:
            return LocalResponse(404, {"detail": f"Not Found: {path}"})
        req_cls, fn = entry
        try:
            req_obj = req_cls.model_validate(json_body or {})
        except ValidationError as exc:
            return LocalResponse(422, {"detail": json.loads(exc.json())})
        try:
            res = fn(req_obj)
            return LocalResponse(200, json.loads(res.model_dump_json()))
        except ValueError as exc:
            return LocalResponse(400, {"detail": str(exc)})


_POST_ROUTES: dict[str, tuple[type, Callable[[Any], Any]]] = {
    "/api/wp1/simulate": (SimulationRequest, run_simulation),
    "/api/wp1/infer": (InferenceRequest, run_inference),
    "/api/wp2/solve": (SolverRequest, run_solver),
    "/api/wp3/benchmark": (BenchmarkRequest, run_benchmark),
    "/api/pipeline/run": (FullStudyRequest, run_full_study),
    "/api/empirical/run": (EmpiricalRunRequest, run_empirical_study),
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
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/api/health":
            self._send_json(200, get_health().model_dump())
        elif path == "/api/defaults":
            self._send_json(200, get_defaults().model_dump())
        elif path == "/api/empirical/catalog":
            ims_dir = qs.get("ims_dir", [None])[0]
            pron_dir = qs.get("pronostia_dir", [None])[0]
            cache_dir = qs.get("cache_dir", ["results/cache"])[0]
            self._send_json(200, get_empirical_catalog(ims_dir=ims_dir, pronostia_dir=pron_dir, cache_dir=cache_dir).model_dump())
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
        except ValueError as exc:
            self._send_json(400, {"detail": str(exc)})
        except Exception as exc:
            self._send_json(500, {"detail": str(exc)})


def create_app():
    if not HAS_FASTAPI:
        return LocalAPITestClient()

    application = FastAPI(
        title="IMDP-PHM Scientific Decision API",
        description="Condition-Based Maintenance under Imprecise Self-Exciting Jump-Diffusion Degradation",
        version=__version__,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health", response_model=HealthResponse, tags=["System"])
    def api_health() -> HealthResponse:
        return get_health()

    @application.get("/api/defaults", response_model=DefaultsResponse, tags=["System"])
    def api_defaults() -> DefaultsResponse:
        return get_defaults()

    @application.get("/api/empirical/catalog", response_model=EmpiricalCatalogResponse, tags=["Empirical"])
    def api_empirical_catalog(
        ims_dir: Optional[str] = None,
        pronostia_dir: Optional[str] = None,
        cache_dir: str = "results/cache",
    ) -> EmpiricalCatalogResponse:
        return get_empirical_catalog(ims_dir=ims_dir, pronostia_dir=pronostia_dir, cache_dir=cache_dir)

    @application.post("/api/empirical/run", response_model=EmpiricalRunResponse, tags=["Empirical"])
    def api_empirical_run(req: EmpiricalRunRequest) -> EmpiricalRunResponse:
        return run_empirical_study(req)

    @application.post("/api/wp1/simulate", response_model=SimulationResponse, tags=["WP1"])
    def api_wp1_sim(req: SimulationRequest) -> SimulationResponse:
        return run_simulation(req)

    @application.post("/api/wp1/infer", response_model=InferenceResponse, tags=["WP1"])
    def api_wp1_inf(req: InferenceRequest) -> InferenceResponse:
        return run_inference(req)

    @application.post("/api/wp2/solve", response_model=SolverResponse, tags=["WP2"])
    def api_wp2(req: SolverRequest) -> SolverResponse:
        return run_solver(req)

    @application.post("/api/wp3/benchmark", response_model=BenchmarkResponse, tags=["WP3"])
    def api_wp3(req: BenchmarkRequest) -> BenchmarkResponse:
        return run_benchmark(req)

    @application.post("/api/pipeline/run", response_model=FullStudyResponse, tags=["Pipeline"])
    def api_pipeline(req: FullStudyRequest) -> FullStudyResponse:
        return run_full_study(req)

    return application


app = create_app()


def create_fallback_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), IMDPRequestHandler)

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_FILE)

logger = logging.getLogger("trading_bot.otel")


def _clean(value: Optional[str]) -> str:
    return (value or "").strip().strip('"').strip("'")


def configure_langsmith() -> bool:
    """Enable LangSmith from .env. No-op under pytest so tests stay offline."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    key = _clean(os.getenv("LANGSMITH_API_KEY"))
    flag = _clean(os.getenv("LANGSMITH_TRACING")).lower()
    if not key or flag not in {"1", "true", "yes", "on"}:
        return False
    project = _clean(os.getenv("LANGSMITH_PROJECT")) or "trading-bot-v2"
    endpoint = _clean(os.getenv("LANGSMITH_ENDPOINT")) or "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    workspace = _clean(os.getenv("LANGSMITH_WORKSPACE_ID"))
    if workspace:
        os.environ["LANGSMITH_WORKSPACE_ID"] = workspace
    return True


LANGSMITH_ON = configure_langsmith()


def tracing_enabled() -> bool:
    global LANGSMITH_ON
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    if not LANGSMITH_ON:
        LANGSMITH_ON = configure_langsmith()
    return LANGSMITH_ON


class Span:
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.started = time.perf_counter()
        self.duration_ms: Optional[float] = None
        self.status = "ok"
        self.error: Optional[str] = None

    def finish(self, status: str = "ok", error: Optional[str] = None):
        self.duration_ms = (time.perf_counter() - self.started) * 1000
        self.status = status
        self.error = error
        logger.info(
            "span name=%s status=%s duration_ms=%.2f attrs=%s error=%s",
            self.name,
            self.status,
            self.duration_ms or 0,
            self.attributes,
            self.error,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


def _safe_meta(attributes: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


class Tracer:
    """Local spans plus LangSmith runs when LANGSMITH_TRACING=true and a key is set."""

    def __init__(self):
        self.spans: List[Span] = []

    @contextmanager
    def start_as_current_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name, attributes)
        self.spans.append(span)
        meta = _safe_meta(attributes)
        ctx = nullcontext()
        if tracing_enabled():
            try:
                from langsmith.run_helpers import trace

                ctx = trace(
                    name=name,
                    run_type="chain",
                    inputs=meta,
                    metadata=meta,
                    tags=["trading-bot", "v2"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LangSmith trace context failed: %s", exc)
                ctx = nullcontext()
        try:
            with ctx as run:
                yield span
                if run is not None:
                    try:
                        run.end(outputs={"status": "ok", **meta})
                    except Exception:
                        pass
                span.finish("ok")
        except Exception as exc:
            span.finish("error", str(exc))
            raise

    def p95_ms(self) -> float:
        durations = sorted(s.duration_ms or 0 for s in self.spans if s.duration_ms is not None)
        if not durations:
            return 0.0
        idx = max(int(0.95 * (len(durations) - 1)), 0)
        return durations[idx]


TRACER = Tracer()

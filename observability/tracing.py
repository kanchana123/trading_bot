import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("trading_bot.otel")


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


class Tracer:
    """OpenTelemetry-shaped tracer. Exports to logs; OTLP/LangSmith optional."""

    def __init__(self):
        self.spans: List[Span] = []

    @contextmanager
    def start_as_current_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name, attributes)
        self.spans.append(span)
        try:
            yield span
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

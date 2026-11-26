"""JSON events emitted by the monitoring Flower strategy.

Schema ``fl-event/v1``. Later middleware (phase 2) can tail the JSONL log
and forward these records over WebSocket without changing the strategy.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any

SCHEMA = "fl-event/v1"

# Node statuses the 3D view will use: training while a fit message is in
# flight, uploading once weights come back, online after evaluation.
STATUS_TRAINING = "training"
STATUS_UPLOADING = "uploading"
STATUS_EVALUATING = "evaluating"
STATUS_ONLINE = "online"
STATUS_FAILED = "failed"


def to_jsonable(value: Any) -> Any:
    """Convert Flower / NumPy scalars into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        return to_jsonable(item())
    return str(value)


@dataclass(frozen=True)
class FLEvent:
    """One lifecycle event from the federated round loop."""

    event: str
    server_round: int
    phase: str
    status: str
    node_id: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = to_jsonable(self.metrics)
        return payload


EventHandler = Callable[[FLEvent], None]


class EventLog:
    """Fan-out sink: memory, optional callback, optional JSONL file."""

    def __init__(
        self,
        path: str | Path | None = None,
        handler: EventHandler | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.handler = handler
        self.events: list[FLEvent] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def emit(self, event: FLEvent) -> None:
        self.events.append(event)
        if self.handler is not None:
            self.handler(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

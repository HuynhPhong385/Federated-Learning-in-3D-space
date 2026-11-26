"""FedAvg strategy that hooks round start/end and each client step.

Flower's message-based ``Strategy.start`` loop is:

1. ``configure_train``  -> round starts, selected clients enter training
2. clients run ``@app.train`` (the client step)
3. ``aggregate_train``  -> weights uploaded, FedAvg
4. ``configure_evaluate`` / ``aggregate_evaluate`` -> round ends
"""

from __future__ import annotations

from collections.abc import Iterable
from logging import INFO
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.supercore import log

from fl_core.events import (
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_ONLINE,
    STATUS_TRAINING,
    STATUS_UPLOADING,
    EventHandler,
    EventLog,
    FLEvent,
    to_jsonable,
)


def _node_id(message: Message, *, reply: bool) -> int:
    metadata = message.metadata
    node = metadata.src_node_id if reply else metadata.dst_node_id
    return int(node)


def _metrics_of(message: Message) -> dict:
    if message.has_error():
        reason = message.error.reason if message.error is not None else "unknown"
        return {"error": reason}
    metrics = message.content.get("metrics")
    if metrics is None:
        return {}
    return to_jsonable(dict(metrics.items()))


class MonitoringFedAvg(FedAvg):
    """FedAvg plus a structured event stream for the 3D monitor.

    Parameters
    ----------
    event_log_path:
        JSONL file (``fl-event/v1``) appended during the run. Created empty
        when the strategy is constructed.
    on_event:
        Optional callback invoked for every event, in addition to the file.
    """

    def __init__(
        self,
        event_log_path: str | Path | None = None,
        on_event: EventHandler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.log = EventLog(event_log_path, on_event)
        self._train_metrics: dict = {}
        self._counts = {"train_ok": 0, "train_failed": 0, "evaluate_ok": 0, "evaluate_failed": 0}

    @property
    def events(self) -> list[FLEvent]:
        return list(self.log.events)

    def summary(self) -> None:
        super().summary()
        target = self.log.path if self.log.path is not None else "(memory only)"
        log(INFO, "\t└──> Monitoring events: %s", target)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        self._train_metrics = {}
        self._counts = {"train_ok": 0, "train_failed": 0, "evaluate_ok": 0, "evaluate_failed": 0}
        self.log.emit(
            FLEvent(
                event="round_start",
                server_round=server_round,
                phase="train",
                status="started",
            )
        )
        messages = list(super().configure_train(server_round, arrays, config, grid))
        for message in messages:
            self.log.emit(
                FLEvent(
                    event="client_step",
                    server_round=server_round,
                    phase="train",
                    status=STATUS_TRAINING,
                    node_id=_node_id(message, reply=False),
                )
            )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        for message in reply_list:
            failed = message.has_error()
            self._counts["train_failed" if failed else "train_ok"] += 1
            self.log.emit(
                FLEvent(
                    event="client_step",
                    server_round=server_round,
                    phase="train",
                    status=STATUS_FAILED if failed else STATUS_UPLOADING,
                    node_id=_node_id(message, reply=True),
                    metrics=_metrics_of(message),
                )
            )
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        self._train_metrics = to_jsonable(dict(metrics.items())) if metrics is not None else {}
        return arrays, metrics

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        for message in messages:
            self.log.emit(
                FLEvent(
                    event="client_step",
                    server_round=server_round,
                    phase="evaluate",
                    status=STATUS_EVALUATING,
                    node_id=_node_id(message, reply=False),
                )
            )
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        reply_list = list(replies)
        for message in reply_list:
            failed = message.has_error()
            self._counts["evaluate_failed" if failed else "evaluate_ok"] += 1
            self.log.emit(
                FLEvent(
                    event="client_step",
                    server_round=server_round,
                    phase="evaluate",
                    status=STATUS_FAILED if failed else STATUS_ONLINE,
                    node_id=_node_id(message, reply=True),
                    metrics=_metrics_of(message),
                )
            )
        metrics = super().aggregate_evaluate(server_round, reply_list)
        evaluate_metrics = to_jsonable(dict(metrics.items())) if metrics is not None else {}
        self.log.emit(
            FLEvent(
                event="round_end",
                server_round=server_round,
                phase="round",
                status="completed",
                metrics={
                    **{f"train_{key}": value for key, value in self._train_metrics.items()},
                    **{f"evaluate_{key}": value for key, value in evaluate_metrics.items()},
                    **self._counts,
                },
            )
        )
        return metrics

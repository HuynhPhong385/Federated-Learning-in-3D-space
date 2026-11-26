"""Hooks on MonitoringFedAvg: round start, client step, round end."""

import json

import numpy as np
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from fl_core.strategy import MonitoringFedAvg


@pytest.fixture(autouse=True)
def _flower_task_identity():
    """Message construction in Flower 1.45 requires a process-level task identity."""
    TaskIdentity.run_id = 1
    TaskIdentity.task_id = 1
    TaskIdentity.node_id = 0
    yield
    TaskIdentity.run_id = None
    TaskIdentity.task_id = None
    TaskIdentity.node_id = None


class _Grid:
    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = node_ids

    def get_node_ids(self) -> list[int]:
        return list(self._node_ids)


def _reply(node_id: int, weights: list[np.ndarray], loss: float, accuracy: float, num_examples: int) -> Message:
    outgoing = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(weights),
                "config": ConfigRecord({"lr": 0.1, "local-epochs": 1}),
            }
        ),
        message_type=MessageType.TRAIN,
        dst_node_id=node_id,
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(weights),
                "metrics": MetricRecord(
                    {
                        "loss": loss,
                        "accuracy": accuracy,
                        "num-examples": num_examples,
                    }
                ),
            }
        ),
        reply_to=outgoing,
    )


def _eval_reply(node_id: int, loss: float, accuracy: float, num_examples: int) -> Message:
    outgoing = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord([np.zeros(2), np.zeros(1)]),
                "config": ConfigRecord({}),
            }
        ),
        message_type=MessageType.EVALUATE,
        dst_node_id=node_id,
    )
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "loss": loss,
                        "accuracy": accuracy,
                        "num-examples": num_examples,
                    }
                )
            }
        ),
        reply_to=outgoing,
    )


def test_round_start_selects_each_client(tmp_path):
    strategy = MonitoringFedAvg(event_log_path=tmp_path / "events.jsonl", min_available_nodes=2)
    messages = list(
        strategy.configure_train(
            server_round=1,
            arrays=ArrayRecord([np.zeros(2), np.zeros(1)]),
            config=ConfigRecord({"lr": 0.2, "local-epochs": 1}),
            grid=_Grid([10, 11, 12]),
        )
    )

    assert messages
    names = [(event.event, event.status) for event in strategy.events]
    assert names[0] == ("round_start", "started")
    training = [event for event in strategy.events if event.event == "client_step"]
    assert {event.node_id for event in training} == {message.metadata.dst_node_id for message in messages}
    assert all(event.status == "training" and event.server_round == 1 for event in training)

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["schema"] == "fl-event/v1"


def test_client_step_and_round_end_keep_fedavg():
    strategy = MonitoringFedAvg(min_available_nodes=2)
    list(
        strategy.configure_train(
            1,
            ArrayRecord([np.zeros(2), np.zeros(1)]),
            ConfigRecord({"lr": 0.1, "local-epochs": 1}),
            _Grid([1, 2]),
        )
    )
    left = [np.array([1.0, 0.0]), np.array([0.0])]
    right = [np.array([0.0, 1.0]), np.array([2.0])]
    arrays, train_metrics = strategy.aggregate_train(
        1,
        [
            _reply(1, left, loss=0.4, accuracy=0.5, num_examples=1),
            _reply(2, right, loss=0.2, accuracy=1.0, num_examples=3),
        ],
    )
    evaluate_metrics = strategy.aggregate_evaluate(
        1,
        [
            _eval_reply(1, loss=0.5, accuracy=0.5, num_examples=1),
            _eval_reply(2, loss=0.1, accuracy=1.0, num_examples=3),
        ],
    )

    merged = arrays.to_numpy_ndarrays()
    np.testing.assert_allclose(merged[0], np.array([0.25, 0.75]))
    np.testing.assert_allclose(merged[1], np.array([1.5]))
    assert train_metrics["loss"] == (0.4 * 1 + 0.2 * 3) / 4
    assert evaluate_metrics["accuracy"] == (0.5 * 1 + 1.0 * 3) / 4

    uploaded = [
        event
        for event in strategy.events
        if event.event == "client_step" and event.phase == "train" and event.status == "uploading"
    ]
    assert [event.node_id for event in uploaded] == [1, 2]
    assert uploaded[0].metrics["loss"] == 0.4

    ended = strategy.events[-1]
    assert ended.event == "round_end"
    assert ended.metrics["train_ok"] == 2
    assert ended.metrics["evaluate_ok"] == 2
    assert ended.metrics["train_loss"] == train_metrics["loss"]
    assert ended.metrics["evaluate_accuracy"] == evaluate_metrics["accuracy"]

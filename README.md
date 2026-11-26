# Federated-Learning-in-3D-space

Platform for designing and monitoring federated learning in 3D space (Nhóm 10, SGOD).

## Phase 1 — monitoring strategy

`MonitoringFedAvg` (`fl_core/strategy.py`) subclasses Flower `FedAvg` and emits `fl-event/v1` records:

| Hook | When | Event |
|---|---|---|
| Round start | `configure_train` | `round_start` |
| Client step | client selected, weights returned, evaluation | `client_step` (`training` / `uploading` / `evaluating` / `online` / `failed`) |
| Round end | `aggregate_evaluate` | `round_end` with aggregated loss and accuracy |

Events stay in memory, go to an optional callback, and append to `artifacts/fl_events.jsonl` for the phase-2 WebSocket middleware.

Requires Python 3.11.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
flwr run . --stream
```

"""ServerApp entry point. Runs MonitoringFedAvg for the configured rounds."""

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from fl_app.task import initial_model
from fl_core import MonitoringFedAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Start federated averaging and write round/client events to JSONL."""
    run_config = context.run_config
    num_rounds = int(run_config["num-server-rounds"])
    strategy = MonitoringFedAvg(
        event_log_path=str(run_config.get("event-log", "artifacts/fl_events.jsonl")),
        min_available_nodes=2,
    )
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial_model()),
        num_rounds=num_rounds,
        train_config=ConfigRecord(
            {
                "lr": float(run_config["lr"]),
                "local-epochs": int(run_config["local-epochs"]),
            }
        ),
        evaluate_config=ConfigRecord({}),
    )

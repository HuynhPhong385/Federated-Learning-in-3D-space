"""ClientApp: one local train or evaluate step per incoming message."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_app.task import evaluate_model, local_dataset, train_local

app = ClientApp()


def _partition(context: Context) -> tuple[int, int]:
    node_config = context.node_config
    partition_id = int(node_config.get("partition-id", 0))
    num_examples = int(context.run_config.get("num-examples", 64))
    return partition_id, num_examples


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Client step: fit the global arrays on this node's synthetic partition."""
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    config = msg.content["config"]
    partition_id, num_examples = _partition(context)
    features, labels = local_dataset(partition_id, num_examples)
    updated, loss, accuracy = train_local(
        arrays,
        features,
        labels,
        lr=float(config["lr"]),
        epochs=int(config["local-epochs"]),
    )
    metrics = MetricRecord(
        {
            "loss": loss,
            "accuracy": accuracy,
            "num-examples": num_examples,
            "partition-id": partition_id,
        }
    )
    content = RecordDict({"arrays": ArrayRecord(updated), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Score the current global model. Evaluation does not change the arrays."""
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    partition_id, num_examples = _partition(context)
    features, labels = local_dataset(partition_id, num_examples)
    loss, accuracy = evaluate_model(arrays[0], arrays[1], features, labels)
    metrics = MetricRecord(
        {
            "loss": loss,
            "accuracy": accuracy,
            "num-examples": num_examples,
            "partition-id": partition_id,
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

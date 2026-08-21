from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from ...logger import logger
from ...schema import TrainingMode
from ...transport.base import Message
from .protocol import _extract_payload


@torch.no_grad()
def evaluate_client(client, round: int = 0) -> dict[str, float | int]:
    """Evaluate a client model and optionally persist per-sample metrics."""
    if len(client.dataset.test_dataset) == 0:
        logger.warning(
            "Client %s: test dataset is empty — returning zero metrics.",
            client.client_id,
        )
        return _empty_metrics()

    client.model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []

    for step, (x, y) in enumerate(client.test_loader, start=1):
        x = x.to(client.device, non_blocking=True)
        y = y.to(client.device, non_blocking=True)

        if client.mode is TrainingMode.federated:
            logits = client.model(x).reshape(-1)
        else:
            activations = client.model(x)
            request = Message(
                type="eval_step",
                sender=client.client_id,
                round=round,
                step=step,
                payload={
                    "activations": activations.cpu(),
                    "labels": y.cpu(),
                },
            )
            client.to_server.send(request)
            response = client.from_server.recv()
            logits = _extract_payload(
                response,
                "logits",
                "logits",
                client.client_id,
                round,
                step,
                request.request_id,
            )
            if logits is None:
                raise RuntimeError(
                    f"Client {client.client_id}: invalid split evaluation "
                    f"response for round={round} step={step}"
                )
            logits = logits.to(client.device).reshape(-1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        y = y.reshape(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    if total == 0:
        logger.warning("Client %s: no samples evaluated.", client.client_id)
        return _empty_metrics()

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    all_probs_np = torch.cat(all_probs).numpy()
    metrics_path = getattr(client, "metrics_path", None)
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "probs": all_probs_np,
                "preds": all_preds_np,
                "labels": all_labels_np,
            }
        ).to_csv(
            metrics_path / f"Client{client.client_id}_round_{round}_eval.csv",
            index=False,
        )

    return {
        "accuracy": correct / total,
        "f1": float(
            f1_score(
                all_labels_np,
                all_preds_np,
                average="binary",
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                all_labels_np,
                all_preds_np,
                average="binary",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                all_labels_np,
                all_preds_np,
                average="binary",
                zero_division=0,
            )
        ),
        "num_samples": total,
        "num_positive_labels": int(all_labels_np.sum()),
        "num_positive_predictions": int(all_preds_np.sum()),
    }


def _empty_metrics() -> dict[str, float | int]:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "num_samples": 0,
        "num_positive_labels": 0,
        "num_positive_predictions": 0,
    }

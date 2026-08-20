"""Model forward, backward and evaluation operations for SplitServer."""

import torch
import torch.nn as nn

from ...logger import logger
from ...model.server_side_model import ServerSideModel
from ...transport.base import Channel, Message


def _handle_train_batch(
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    client_channels: dict[str, dict[str, Channel]],
    parallel: bool = True,
) -> float | None:
    """Train one combined or sequential server-side split batch."""
    model.train()
    if parallel:
        result = _forward_parallel(batch_msgs, model, criterion, device)
    else:
        result = _forward_sequential(
            batch_msgs, model, criterion, device, len(batch_msgs)
        )
    if result is None:
        return None
    grads_per_client, batch_loss = result
    for client_id, message in batch_msgs.items():
        client_channels[client_id]["downlink"].send(
            Message(
                type="gradients",
                sender="split_server",
                round=message.round,
                step=message.step,
                request_id=message.request_id,
                payload={"gradients": grads_per_client[client_id]},
            )
        )
    return batch_loss


def _forward_parallel(
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float] | None:
    """Run one concatenated forward/backward and split activation gradients."""
    activations_list, labels_list, sizes, client_order = [], [], [], []
    for client_id in sorted(batch_msgs):
        message = batch_msgs[client_id]
        activations = message.payload["activations"].to(device)
        labels = message.payload["labels"].to(device).float()
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        sizes.append(activations.shape[0])
        client_order.append(client_id)
        activations_list.append(activations)
        labels_list.append(labels)

    combined = torch.cat(activations_list, dim=0).detach().requires_grad_(True)
    expected = torch.cat(labels_list, dim=0)
    outputs = model(combined)
    if outputs.shape != expected.shape:
        logger.error(
            "SplitServer [parallel]: shape mismatch %s vs %s — "
            "aborting batch.",
            outputs.shape,
            expected.shape,
        )
        return None
    loss = criterion(outputs, expected)
    loss_value = loss.item()
    loss.backward()
    if combined.grad is None:
        logger.error(
            "SplitServer [parallel]: H.grad is None — sending zero gradients."
        )
        gradients = torch.zeros_like(combined)
    else:
        gradients = combined.grad.detach()
    splits = torch.split(gradients, sizes, dim=0)
    return {
        client_id: gradient.cpu()
        for client_id, gradient in zip(client_order, splits, strict=True)
    }, loss_value


def _forward_sequential(
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    n_clients: int,
) -> tuple[dict[str, torch.Tensor], float] | None:
    """Run client forwards sequentially while accumulating model gradients."""
    gradients: dict[str, torch.Tensor] = {}
    total_loss = 0.0
    for client_id, message in batch_msgs.items():
        activations = message.payload["activations"].to(device)
        labels = message.payload["labels"].to(device).float()
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        detached = activations.detach().requires_grad_(True)
        outputs = model(detached)
        if outputs.shape != labels.shape:
            logger.error(
                "SplitServer [sequential]: shape mismatch for client %s "
                "%s vs %s — aborting batch.",
                client_id,
                outputs.shape,
                labels.shape,
            )
            model.zero_grad()
            return None
        loss = criterion(outputs, labels) / n_clients
        total_loss += loss.item()
        loss.backward()
        if detached.grad is None:
            logger.error(
                "SplitServer [sequential]: H.grad is None for client %s "
                "— sending zero gradients.",
                client_id,
            )
            gradients[client_id] = torch.zeros_like(activations)
        else:
            gradients[client_id] = detached.grad.detach().cpu()
    return gradients, total_loss


def _handle_eval_single(
    message: Message,
    client_id: str,
    model: nn.Module,
    device: torch.device,
    client_channels: dict[str, dict[str, Channel]],
    all_eval_probs: list,
    all_eval_labels: list,
) -> None:
    """Evaluate one client batch with training noise and gradients disabled."""
    model.eval()
    activations = message.payload["activations"].to(device)
    labels = message.payload["labels"].to(device).float().reshape(-1)
    with torch.no_grad():
        logits = model(activations).reshape(-1)
        probabilities = torch.sigmoid(logits)
    all_eval_probs.append(probabilities.cpu())
    all_eval_labels.append(labels.cpu())
    client_channels[client_id]["downlink"].send(
        Message(
            type="logits",
            sender="split_server",
            round=message.round,
            step=message.step,
            request_id=message.request_id,
            payload={"logits": logits.cpu()},
        )
    )

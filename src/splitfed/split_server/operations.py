"""Model forward, backward and evaluation operations for SplitServer."""

import torch
import torch.nn as nn

from ...logger import logger
from ...model.server_side_model import ServerSideModel
from ...transport.base import Channel, Message


def _handle_train_concat(
    batch_msgs: dict[str, Message],
    model: ServerSideModel,
    criterion: nn.Module,
    device: torch.device,
    client_channels: dict[str, dict[str, Channel]],
) -> float | None:
    """Train one server batch formed by concatenating client activations."""
    model.train()
    result = _forward_concat(batch_msgs, model, criterion, device)
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


def _forward_concat(
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
            "SplitServer [concat]: shape mismatch %s vs %s — "
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
            "SplitServer [concat]: H.grad is None — sending zero gradients."
        )
        gradients = torch.zeros_like(combined)
    else:
        gradients = combined.grad.detach()
    splits = torch.split(gradients, sizes, dim=0)
    return {
        client_id: gradient.cpu()
        for client_id, gradient in zip(client_order, splits, strict=True)
    }, loss_value


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

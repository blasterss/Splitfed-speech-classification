import queue

import torch
import torch.multiprocessing as mp

from src.model.client_side_model import ClientSideModel
from src.model.speech_model import SpeechRecognitionModel
from src.schema import FedServerConfig, SplitServerConfig
from src.splitfed.client import _extract_payload, _validate_global_update
from src.splitfed.fed_server import _fed_server_worker
from src.splitfed.split_server import (
    _split_server_worker_batch,
    _split_server_worker_personalized,
)
from src.transport.message import Message
from src.utils.state import deserialize_state_dict


class SpawnQueueChannel:
    def __init__(self, context, timeout=10):
        self.queue = context.Queue(maxsize=8)
        self.timeout = timeout

    def send(self, message):
        self.queue.put(message, timeout=self.timeout)

    def recv(self):
        return self.queue.get(timeout=self.timeout)

    def recv_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


def _synthetic_client_worker(
    client_id,
    local_steps,
    split_uplink,
    split_downlink,
    fed_uplink,
    fed_downlink,
    result_queue,
):
    torch.set_num_threads(1)
    torch.manual_seed(100 + local_steps)
    model = ClientSideModel(input_channels=3, noise=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for step in range(1, local_steps + 1):
        inputs = torch.randn(1, 3, 64)
        labels = torch.tensor([float(step % 2)])
        activations = model(inputs)
        request = Message(
            type="train_step",
            sender=client_id,
            round=1,
            step=step,
            payload={
                "activations": activations.detach(),
                "labels": labels,
            },
        )
        split_uplink.send(request)
        response = split_downlink.recv()
        gradients = _extract_payload(
            response,
            "gradients",
            "gradients",
            client_id,
            1,
            step,
            request.request_id,
        )
        if gradients is None:
            raise RuntimeError("Synthetic client received invalid gradients")
        activations.backward(gradients)
        optimizer.step()
        optimizer.zero_grad()

    split_uplink.send(
        Message(
            type="round_end",
            sender=client_id,
            round=1,
            step=local_steps + 1,
            payload={},
        )
    )
    fed_request = Message(
        type="client_update",
        sender=client_id,
        round=1,
        step=1,
        payload={
            "state_dict": model.state_dict(),
            "dataset_size": local_steps,
        },
    )
    fed_uplink.send(fed_request)
    global_update = fed_downlink.recv()
    state_dict = _validate_global_update(
        global_update, model.state_dict(), 1, fed_request.request_id
    )
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        activations = model(torch.zeros(1, 3, 64))
    eval_request = Message(
        type="eval_step",
        sender=client_id,
        round=1,
        step=1,
        payload={"activations": activations, "labels": torch.zeros(1)},
    )
    split_uplink.send(eval_request)
    logits = _extract_payload(
        split_downlink.recv(),
        "logits",
        "logits",
        client_id,
        1,
        1,
        eval_request.request_id,
    )
    if logits is None or logits.shape != torch.Size([1]):
        raise RuntimeError("Synthetic client received invalid logits")
    result_queue.put((client_id, local_steps))


def _synthetic_federated_client_worker(
    client_id,
    fed_uplink,
    fed_downlink,
    result_queue,
):
    torch.set_num_threads(1)
    torch.manual_seed(200 + int(client_id[-1]))
    model = SpeechRecognitionModel(
        input_channels=3,
        server_side_model_type="cnn_gap",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    logits = model(torch.randn(1, 3, 64))
    loss = criterion(logits, torch.tensor([[1.0]]))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    request = Message(
        type="client_update",
        sender=client_id,
        round=1,
        step=1,
        payload={"state_dict": model.state_dict(), "dataset_size": 1},
    )
    fed_uplink.send(request)
    update = fed_downlink.recv()
    model.load_state_dict(
        _validate_global_update(
            update, model.state_dict(), 1, request.request_id
        )
    )
    model.eval()
    with torch.no_grad():
        eval_logits = model(torch.zeros(1, 3, 64))
    result_queue.put((client_id, tuple(eval_logits.shape)))


def _synthetic_personalized_client_worker(
    client_id,
    local_steps,
    split_uplink,
    split_downlink,
    result_queue,
):
    torch.set_num_threads(1)
    torch.manual_seed(300 + local_steps)
    model = ClientSideModel(input_channels=3, noise=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for step in range(1, local_steps + 1):
        activations = model(torch.randn(1, 3, 64))
        request = Message(
            type="train_step",
            sender=client_id,
            round=1,
            step=step,
            payload={
                "activations": activations.detach(),
                "labels": torch.tensor([float(step % 2)]),
            },
        )
        split_uplink.send(request)
        gradients = _extract_payload(
            split_downlink.recv(),
            "gradients",
            "gradients",
            client_id,
            1,
            step,
            request.request_id,
        )
        if gradients is None:
            raise RuntimeError("Missing personalized split gradients")
        activations.backward(gradients)
        optimizer.step()
        optimizer.zero_grad()
    split_uplink.send(
        Message(
            type="round_end",
            sender=client_id,
            round=1,
            step=local_steps + 1,
            payload={},
        )
    )
    result_queue.put((client_id, local_steps))


def test_spawned_splitfed_training_cycle_with_unequal_client_steps():
    context = mp.get_context("spawn")
    stop_event = context.Event()
    split_result_queue = context.Queue(maxsize=1)
    fed_result_queue = context.Queue(maxsize=1)
    client_result_queue = context.Queue(maxsize=2)
    client_ids = ("client-0", "client-1")
    channels = {}
    for client_id in client_ids:
        channels[client_id] = {
            "split_uplink": SpawnQueueChannel(context),
            "split_downlink": SpawnQueueChannel(context),
            "federated_uplink": SpawnQueueChannel(context),
            "federated_downlink": SpawnQueueChannel(context),
        }

    split_config = SplitServerConfig(
        model={
            "pos_weight": 1,
            "optimizer": "adam",
            "lr": 0.001,
            "device": "cpu",
            "gradient_accumulation_steps": 2,
            "batch_timeout_sec": 10,
        },
        seed=42,
        model_scope="shared",
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
    )
    fed_config = FedServerConfig(
        strategy="fedavg",
        seed=42,
        device="cpu",
        aggregation_freq=1,
        min_clients=2,
        quorum_timeout_sec=10,
        federated_uplink_channel="federated_uplink",
        federated_downlink_channel="federated_downlink",
    )
    split_channels = {
        client_id: {
            "uplink": channels[client_id]["split_uplink"],
            "downlink": channels[client_id]["split_downlink"],
        }
        for client_id in client_ids
    }
    fed_channels = {
        client_id: {
            "uplink": channels[client_id]["federated_uplink"],
            "downlink": channels[client_id]["federated_downlink"],
        }
        for client_id in client_ids
    }
    processes = [
        context.Process(
            target=_split_server_worker_batch,
            args=(
                split_config,
                split_channels,
                stop_event,
                split_result_queue,
            ),
            name="SmokeSplitServer",
        ),
        context.Process(
            target=_fed_server_worker,
            args=(fed_config, fed_channels, 2, stop_event, fed_result_queue),
            name="SmokeFedServer",
        ),
    ]
    for client_id, local_steps in zip(client_ids, (1, 2), strict=True):
        client_channels = channels[client_id]
        processes.append(
            context.Process(
                target=_synthetic_client_worker,
                args=(
                    client_id,
                    local_steps,
                    client_channels["split_uplink"],
                    client_channels["split_downlink"],
                    client_channels["federated_uplink"],
                    client_channels["federated_downlink"],
                    client_result_queue,
                ),
                name=f"Smoke-{client_id}",
            )
        )

    try:
        for process in processes:
            process.start()
        for process in processes[2:]:
            process.join(timeout=30)

        assert all(not process.is_alive() for process in processes[2:])
        assert all(process.exitcode == 0 for process in processes[2:])
        results = {client_result_queue.get(timeout=2) for _ in client_ids}
        assert results == {("client-0", 1), ("client-1", 2)}
    finally:
        stop_event.set()
        split_payload = split_result_queue.get(timeout=10)
        fed_payload = fed_result_queue.get(timeout=10)
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)
    assert deserialize_state_dict(split_payload)
    assert deserialize_state_dict(fed_payload)


def test_spawned_federated_cycle_uses_complete_models_without_split_server():
    context = mp.get_context("spawn")
    stop_event = context.Event()
    fed_result_queue = context.Queue(maxsize=1)
    client_result_queue = context.Queue(maxsize=2)
    client_ids = ("client-0", "client-1")
    channels = {
        client_id: {
            "uplink": SpawnQueueChannel(context),
            "downlink": SpawnQueueChannel(context),
        }
        for client_id in client_ids
    }
    fed_config = FedServerConfig(
        strategy="fedavg",
        seed=42,
        device="cpu",
        aggregation_freq=1,
        min_clients=2,
        quorum_timeout_sec=10,
        federated_uplink_channel="federated_uplink",
        federated_downlink_channel="federated_downlink",
    )
    server = context.Process(
        target=_fed_server_worker,
        args=(fed_config, channels, 2, stop_event, fed_result_queue),
        name="FederatedSmokeServer",
    )
    clients = [
        context.Process(
            target=_synthetic_federated_client_worker,
            args=(
                client_id,
                channels[client_id]["uplink"],
                channels[client_id]["downlink"],
                client_result_queue,
            ),
            name=f"FederatedSmoke-{client_id}",
        )
        for client_id in client_ids
    ]

    try:
        server.start()
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=30)
        assert all(not client.is_alive() for client in clients)
        assert all(client.exitcode == 0 for client in clients)
        results = {client_result_queue.get(timeout=2) for _ in clients}
        assert results == {("client-0", (1, 1)), ("client-1", (1, 1))}
    finally:
        stop_event.set()
        fed_payload = fed_result_queue.get(timeout=10)
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join(timeout=5)

    assert not server.is_alive()
    assert server.exitcode == 0
    assert deserialize_state_dict(fed_payload)


def test_spawned_personalized_split_keeps_per_client_server_states():
    context = mp.get_context("spawn")
    stop_event = context.Event()
    server_result_queue = context.Queue(maxsize=1)
    client_result_queue = context.Queue(maxsize=2)
    client_ids = ("client-0", "client-1")
    channels = {
        client_id: {
            "uplink": SpawnQueueChannel(context),
            "downlink": SpawnQueueChannel(context),
        }
        for client_id in client_ids
    }
    config = SplitServerConfig(
        model={
            "pos_weight": 1,
            "optimizer": "adam",
            "lr": 0.001,
            "device": "cpu",
            "gradient_accumulation_steps": 2,
            "batch_timeout_sec": 10,
        },
        seed=42,
        model_scope="personalized",
        split_uplink_channel="split_uplink",
        split_downlink_channel="split_downlink",
    )
    server = context.Process(
        target=_split_server_worker_personalized,
        args=(config, channels, stop_event, server_result_queue),
        name="PersonalizedSmokeServer",
    )
    clients = [
        context.Process(
            target=_synthetic_personalized_client_worker,
            args=(
                client_id,
                local_steps,
                channels[client_id]["uplink"],
                channels[client_id]["downlink"],
                client_result_queue,
            ),
            name=f"PersonalizedSmoke-{client_id}",
        )
        for client_id, local_steps in zip(client_ids, (1, 2), strict=True)
    ]

    try:
        server.start()
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=30)
        assert all(not client.is_alive() for client in clients)
        assert all(client.exitcode == 0 for client in clients)
        assert {client_result_queue.get(timeout=2) for _ in clients} == {
            ("client-0", 1),
            ("client-1", 2),
        }
    finally:
        stop_event.set()
        payload = server_result_queue.get(timeout=10)
        server.join(timeout=10)
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join(timeout=5)

    states = deserialize_state_dict(payload)
    assert not server.is_alive()
    assert server.exitcode == 0
    assert set(states) == set(client_ids)
    assert states["client-0"].keys() == states["client-1"].keys()
    assert any(
        not torch.equal(states["client-0"][key], states["client-1"][key])
        for key in states["client-0"]
    )

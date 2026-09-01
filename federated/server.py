import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import Aggregator, ClientMetrics, FederatedClient, StateDict
from .statistics import ClientRoundStats, RoundStats


DeviceLike = Union[str, torch.device, None]


def _is_cuda_device(device: DeviceLike) -> bool:
    """True iff *device* names a CUDA device.

    The FL runtime picks the device from the Aim/Hydra config
    (``runtime.cpu`` / ``runtime.gpu_id`` -> :func:`federated.device.resolve_device`)
    and hands it to the strategy, which forwards it here.  This function
    honours that choice instead of relying on ``torch.cuda.is_available()``.
    """
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


class FederatedServer:
    """Round-based synchronous FL server.

    Implements the ``ServerExecutes`` loop of the FedAvg pseudo-code:
    each round sample ``m = max(C * K, 1)`` clients, ship the current
    global state, collect their updates and merge them through the
    configured :class:`Aggregator`.  Per-round timing and per-client
    metrics are captured in a :class:`RoundStats` record.

    Client updates within a round are dispatched to a persistent
    :class:`ThreadPoolExecutor`; when CUDA is available each client is
    bound to its own :class:`torch.cuda.Stream` so per-client GPU work
    can overlap.  PyTorch tensor ops release the GIL and streams are
    thread-local, so this parallelises the round without changing any
    client-side logic.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        clients: Iterable[FederatedClient],
        aggregator: Aggregator,
        device: DeviceLike = None,
        max_parallel_clients: int = 5,
    ) -> None:

        self.model_fn = model_fn
        self.clients: List[FederatedClient] = list(clients)
        if not self.clients:
            raise ValueError("At least one client is required.")

        self.aggregator = aggregator
        self._round_counter = 0

        init_model = model_fn()
        self._global_state: StateDict = {
            k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()
        }

        self.device: DeviceLike = device
        self._use_cuda = _is_cuda_device(device)
        self._streams: List[Optional[torch.cuda.Stream]] = (
            [torch.cuda.Stream(device=device) for _ in self.clients]
            if self._use_cuda
            else [None] * len(self.clients)
        )
        n_clients = len(self.clients)
        self._max_parallel = min(
            max_parallel_clients if max_parallel_clients else n_clients,
            n_clients,
        )
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_parallel,
            thread_name_prefix="fl-client",
        )

    @property
    def global_state(self) -> StateDict:
        return self._global_state

    def _run_client(
        self,
        idx: int,
        client: FederatedClient,
    ) -> Tuple[int, StateDict, ClientMetrics, float]:
        stream = self._streams[idx]
        t0 = time.perf_counter()
        if stream is not None:
            with torch.cuda.stream(stream):
                new_state, metrics = client.local_update(self._global_state)
            stream.synchronize()
        else:
            new_state, metrics = client.local_update(self._global_state)
        elapsed = time.perf_counter() - t0
        return idx, new_state, metrics, elapsed

    def run_round(self) -> RoundStats:
        self._round_counter += 1
        round_start = time.perf_counter()

        n = len(self.clients)
        updates: List[Optional[Tuple[StateDict, int]]] = [None] * n
        client_stats: List[Optional[ClientRoundStats]] = [None] * n

        futures = [
            self._pool.submit(self._run_client, i, client)
            for i, client in enumerate(self.clients)
        ]
        for fut in futures:
            idx, new_state, metrics, elapsed = fut.result()
            client = self.clients[idx]
            updates[idx] = (new_state, client.num_samples)
            client_stats[idx] = ClientRoundStats(
                client_id=client.client_id,
                num_samples=client.num_samples,
                train_loss=float(metrics.get("train_loss", float("nan"))),
                num_local_steps=int(metrics.get("num_local_steps", 0)),
                local_update_time_s=elapsed,
            )

        agg_start = time.perf_counter()
        self._global_state = self.aggregator.aggregate(updates)
        agg_elapsed = time.perf_counter() - agg_start

        return RoundStats(
            round_idx=self._round_counter,
            selected_client_ids=[int(client.client_id) for client in self.clients],
            client_stats=client_stats,
            aggregation_time_s=agg_elapsed,
            round_time_s=time.perf_counter() - round_start,
        )

    def fit(
        self,
        num_rounds: int,
        on_round_end: Optional[Callable[[RoundStats, StateDict], None]] = None,
    ) -> List[RoundStats]:
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

        history: List[RoundStats] = []
        for _ in range(num_rounds):
            stats = self.run_round()
            history.append(stats)
            if on_round_end is not None:
                on_round_end(stats, self._global_state)
        return history

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

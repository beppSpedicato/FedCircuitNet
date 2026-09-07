import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import Aggregator, ClientMetrics, FederatedClient, StateDict
from .statistics import ClientRoundStats, RoundStats


DeviceLike = Union[str, torch.device, None]

BYTES_PER_MB = 1024 * 1024

def state_nbytes(state: StateDict) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


def _is_cuda_device(device: DeviceLike) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


class FederatedServer:
    """Round-based synchronous FL server.

    Args:
        model_fn: Zero-arg factory building a fresh, initialised model.
        clients: The full client pool (``K``).
        aggregator: Merges the round's updates into the new global state.
        device: Device the clients train on.
        round_clients: Clients sampled per round (``m``).  Values above
            ``K`` are clamped to ``K``; ``None`` means full participation.
        seed: Seed for the client-selection RNG.  A dedicated generator
            is used so the participation sequence is reproducible no
            matter how much of the global RNG local training consumes.
        max_parallel_clients: Thread-pool width.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        clients: Iterable[FederatedClient],
        aggregator: Aggregator,
        device: DeviceLike = None,
        round_clients: Optional[int] = None,
        seed: Optional[int] = None,
        max_parallel_clients: int = 5,
    ) -> None:

        self.model_fn = model_fn
        self.clients: List[FederatedClient] = list(clients)
        if not self.clients:
            raise ValueError("At least one client is required.")

        self.aggregator = aggregator
        self._round_counter = 0

        n_clients = len(self.clients)
        if round_clients is None:
            self.round_clients = n_clients
        elif round_clients < 1:
            raise ValueError(f"round_clients must be >= 1, got {round_clients}")
        else:
            self.round_clients = min(int(round_clients), n_clients)
            if self.round_clients != round_clients:
                print(
                    f"[server] round_clients={round_clients} exceeds the "
                    f"{n_clients} available clients; clamped to {n_clients} "
                    "(full participation)."
                )

        self._selection_rng = torch.Generator()
        if seed is not None:
            self._selection_rng.manual_seed(int(seed))

        self.selection_counts: List[int] = [0] * n_clients
        self.download_bytes: List[int] = [0] * n_clients
        self.upload_bytes: List[int] = [0] * n_clients

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

    def select_clients(self) -> List[int]:
        perm = torch.randperm(len(self.clients), generator=self._selection_rng)
        return sorted(int(i) for i in perm[: self.round_clients].tolist())

    def run_round(self) -> RoundStats:
        self._round_counter += 1
        round_start = time.perf_counter()

        selected = self.select_clients()
        for idx in selected:
            self.selection_counts[idx] += 1

        down_bytes = state_nbytes(self._global_state)

        updates: List[Tuple[StateDict, int]] = []
        client_stats: List[ClientRoundStats] = []

        futures = [
            self._pool.submit(self._run_client, idx, self.clients[idx])
            for idx in selected
        ]
        for fut in futures:
            idx, new_state, metrics, elapsed = fut.result()
            client = self.clients[idx]
            updates.append((new_state, client.num_samples))

            up_bytes = state_nbytes(new_state)
            self.download_bytes[idx] += down_bytes
            self.upload_bytes[idx] += up_bytes

            client_stats.append(
                ClientRoundStats(
                    client_id=client.client_id,
                    num_samples=client.num_samples,
                    train_loss=float(metrics.get("train_loss", float("nan"))),
                    num_local_steps=int(metrics.get("num_local_steps", 0)),
                    local_update_time_s=elapsed,
                    download_mb=down_bytes / BYTES_PER_MB,
                    upload_mb=up_bytes / BYTES_PER_MB,
                )
            )

        agg_start = time.perf_counter()
        self._global_state = self.aggregator.aggregate(updates)
        agg_elapsed = time.perf_counter() - agg_start

        return RoundStats(
            round_idx=self._round_counter,
            selected_client_ids=[
                int(self.clients[idx].client_id) for idx in selected
            ],
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

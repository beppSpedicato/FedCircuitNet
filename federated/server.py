import time
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch.nn as nn

from .base import Aggregator, FederatedClient, StateDict
from .statistics import ClientRoundStats, RoundStats


class FederatedServer:
    """Round-based synchronous FL server.

    Implements the ``ServerExecutes`` loop of the FedAvg pseudo-code:
    each round sample ``m = max(C * K, 1)`` clients, ship the current
    global state, collect their updates and merge them through the
    configured :class:`Aggregator`.  Per-round timing and per-client
    metrics are captured in a :class:`RoundStats` record.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        clients: Iterable[FederatedClient],
        aggregator: Aggregator,
        seed: int = 42,
    ) -> None:

        self.model_fn = model_fn
        self.clients: List[FederatedClient] = list(clients)
        if not self.clients:
            raise ValueError("At least one client is required.")

        self.aggregator = aggregator
        self._rng = np.random.default_rng(seed)
        self._round_counter = 0

        init_model = model_fn()
        self._global_state: StateDict = {
            k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()
        }

    @property
    def global_state(self) -> StateDict:
        return self._global_state


    def run_round(self) -> RoundStats:
        self._round_counter += 1
        round_start = time.perf_counter()


        updates = []
        client_stats: List[ClientRoundStats] = []
        for client in self.clients:
            t0 = time.perf_counter()
            new_state, metrics = client.local_update(self._global_state)
            elapsed = time.perf_counter() - t0
            updates.append((new_state, client.num_samples))
            client_stats.append(
                ClientRoundStats(
                    client_id=client.client_id,
                    num_samples=client.num_samples,
                    train_loss=float(metrics.get("train_loss", float("nan"))),
                    num_local_steps=int(metrics.get("num_local_steps", 0)),
                    local_update_time_s=elapsed,
                )
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

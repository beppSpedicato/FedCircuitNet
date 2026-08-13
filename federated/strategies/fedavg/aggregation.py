from typing import List

import torch

from ...base import Aggregator, ClientUpdate, StateDict


class WeightedAverageAggregator(Aggregator):
    """FedAvg merging rule: ``w_{t+1} = sum_k (n_k / n) * w_{t+1}^k``.

    Non-floating-point buffers (e.g. BatchNorm's ``num_batches_tracked``
    int64 counter) are not averaged; the value from the first update is
    forwarded instead, matching the convention of most reference
    implementations.
    """

    def aggregate(self, client_updates: List[ClientUpdate]) -> StateDict:
        if not client_updates:
            raise ValueError("aggregate() requires at least one client update.")

        total = sum(n for _, n in client_updates)
        if total <= 0:
            raise ValueError(f"Total sample count must be positive, got {total}.")

        first_state = client_updates[0][0]
        aggregated: StateDict = {}

        for key, first_tensor in first_state.items():
            if not first_tensor.is_floating_point():
                aggregated[key] = first_tensor.clone()
                continue

            acc = torch.zeros(first_tensor.shape, dtype=torch.float64)
            for state, n_k in client_updates:
                acc.add_(state[key].to(torch.float64), alpha=n_k / total)
            aggregated[key] = acc.to(first_tensor.dtype)

        return aggregated

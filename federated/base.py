from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch


StateDict = Dict[str, torch.Tensor]
ClientMetrics = Dict[str, float]
ClientUpdate = Tuple[StateDict, int]


class FederatedClient(ABC):
    """Abstract federated client.

    A client owns a local dataset and knows how to derive a new model state
    from a global one.  Strategy-specific behaviour (plain local SGD for
    FedAvg, proximal term for FedProx, control variates for SCAFFOLD, ...)
    is implemented in the concrete subclass.
    """

    client_id: int

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Size of the client's local dataset (n_k in the FedAvg paper)."""
        ...

    @abstractmethod
    def local_update(
        self, global_state: StateDict
    ) -> Tuple[StateDict, ClientMetrics]:
        """Return the updated state and per-round training metrics.

        Args:
            global_state: Parameters broadcast by the server at round start.

        Returns:
            Tuple ``(new_state, metrics)`` where ``metrics`` at minimum
            reports the mean local training loss under key ``"train_loss"``.
        """
        ...


class Aggregator(ABC):
    """Server-side merging strategy.

    Combines the per-client updates collected in one round into a single
    global state.  FedAvg uses a sample-size-weighted average; other
    aggregators (median, trimmed mean, FedBN's BN-skipping variant, ...)
    plug in here without touching client or server code.
    """

    @abstractmethod
    def aggregate(self, client_updates: List[ClientUpdate]) -> StateDict:
        """Merge a list of ``(state_dict, n_samples)`` pairs into one state."""
        ...

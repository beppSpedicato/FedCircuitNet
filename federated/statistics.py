"""Round-level and history-level statistics collected during FL training.

These dataclasses are the primary artefact of :meth:`FedAvgStrategy.train`
and provide a stable schema for downstream analysis (per-client loss traces,
sampling rates, global-model metrics per round, wall-clock timing).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClientRoundStats:
    """Metrics reported by one client for one round of training."""

    client_id: int
    num_samples: int
    train_loss: float
    num_local_steps: int
    local_update_time_s: float


@dataclass
class RoundStats:
    """Metrics for a single FL communication round."""

    round_idx: int
    selected_client_ids: List[int]
    client_stats: List[ClientRoundStats]
    aggregation_time_s: float
    round_time_s: float
    global_metrics: Optional[Dict[str, float]] = None


@dataclass
class TrainingHistory:
    """Full history returned by :meth:`FedAvgStrategy.train`."""

    partition_sizes: List[int] = field(default_factory=list)
    rounds: List[RoundStats] = field(default_factory=list)

    def append_round(self, stats: RoundStats) -> None:
        self.rounds.append(stats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_sizes": list(self.partition_sizes),
            "rounds": [asdict(r) for r in self.rounds],
        }

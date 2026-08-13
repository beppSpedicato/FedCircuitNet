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

    def to_dataframe(self):
        """Return a long-format DataFrame with one row per (round, client)."""
        import pandas as pd

        records: List[Dict[str, Any]] = []
        for r in self.rounds:
            global_metrics = r.global_metrics or {}
            for c in r.client_stats:
                records.append(
                    {
                        "round": r.round_idx,
                        "client_id": c.client_id,
                        "num_samples": c.num_samples,
                        "train_loss": c.train_loss,
                        "num_local_steps": c.num_local_steps,
                        "local_update_time_s": c.local_update_time_s,
                        "aggregation_time_s": r.aggregation_time_s,
                        "round_time_s": r.round_time_s,
                        **{f"global_{k}": v for k, v in global_metrics.items()},
                    }
                )
        return pd.DataFrame.from_records(records)

    def save_json(self, path: str) -> None:
        import json

        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

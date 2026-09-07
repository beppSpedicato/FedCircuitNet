from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ClientRoundStats:
    """Metrics reported by one client for one round of training."""

    client_id: int
    num_samples: int
    train_loss: float
    num_local_steps: int
    local_update_time_s: float
    download_mb: float = 0.0
    upload_mb: float = 0.0


@dataclass
class RoundStats:
    """Metrics for a single FL communication round."""

    round_idx: int
    selected_client_ids: List[int]
    client_stats: List[ClientRoundStats]
    aggregation_time_s: float
    round_time_s: float
    global_metrics: Optional[Dict[str, float]] = None

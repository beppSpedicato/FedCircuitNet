from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from ...base import ClientMetrics, FederatedClient, StateDict
from tqdm import tqdm
from utils import CosineRestartLr


class FedAvgClient(FederatedClient):
    """Client that runs plain local SGD, as in McMahan et al. 2017.

    Corresponds to the ``ClientUpdate(k, w)`` routine of the FedAvg
    pseudo-code: load the broadcast weights, iterate ``E`` local epochs
    over minibatches of size ``B`` with learning rate ``eta``, return the
    updated weights.

    The minibatch size ``B`` is set by the ``DataLoader``'s ``batch_size``;
    passing a loader with ``batch_size=len(dataset)`` recovers the
    full-batch FedSGD special case.
    """

    def __init__(
        self,
        client_id: int,
        model_fn: Callable[[], nn.Module],
        data_loader: DataLoader,
        local_epochs: int,
        learning_rate: float,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        self.client_id = client_id
        self.model_fn = model_fn
        self.data_loader = data_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.device = device
        self._num_samples = len(data_loader.dataset)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def local_update(
        self, global_state: StateDict
    ) -> Tuple[StateDict, ClientMetrics]:
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_state)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate,  betas=(0.9, 0.999), weight_decay=0.0001)
        cosine_lr = CosineRestartLr(self.learning_rate, [self.local_epochs], [1], 1e-7)
        cosine_lr.set_init_lr(optimizer)

        loss_sum = 0.0
        n_steps = 0
    
        with tqdm(total=self.local_epochs) as bar:
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                regular_lr = cosine_lr.get_regular_lr(n_steps)
                cosine_lr._set_lr(optimizer, regular_lr)
                
                optimizer.zero_grad()
                loss = self.loss_fn(model(inputs), targets)
                loss_sum += float(loss.item())
                loss.backward()
                optimizer.step()
                n_steps += 1
                bar.update(1)

                if n_steps % self.local_epochs == 0:
                    break

        new_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        metrics: ClientMetrics = {
            "train_loss": loss_sum / n_steps if n_steps > 0 else float("nan"),
            "num_local_steps": float(n_steps),
        }
        return new_state, metrics

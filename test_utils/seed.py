import numpy as np
import random
import torch


def set_random_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda is True:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    return

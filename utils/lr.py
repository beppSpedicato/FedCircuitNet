from math import cos, pi

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


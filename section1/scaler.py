import torch
import logging
import wandb


log = logging.getLogger(__name__)


class CustomGradScaler:
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000, fixed=False):
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

        self.fixed = fixed
        # флаг, который помогает вызывать scale и unscale_ по-очереди 
        self.scaled_state = False
        # флаг про то, нашли ли мы в unscale_ плохие значения в градиентах
        self.found_bad_param = False
        # количество последовательных нескипнутых итераций обучения
        self.num_unskipped = 0

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.scaled_state:
            return outputs
        else:
            self.scaled_state = True
            self.found_bad_param = False
            return outputs * self.scale_factor

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaled_state:
            self.scaled_state = False
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.grad /= self.scale_factor
                    self.found_bad_param = self.found_bad_param or not torch.isfinite(param.grad).all()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscale_(optimizer)

        if not self.found_bad_param:
            optimizer.step()
    
    def update(self) -> None:
        wandb.log({"section1/gradient scaling factor": self.scale_factor}, step=wandb.run.step)

        # если используем статический скейлинг, то просто ничего не делаем
        if self.fixed:
            return

        if self.found_bad_param:
            self.scale_factor *= self.backoff_factor
            self.num_unskipped = 0
            log.info(f"Decreasing scale factor to {self.scale_factor:.3f}")
        else:
            self.num_unskipped += 1
            if self.num_unskipped == self.growth_interval:
                self.scale_factor *= self.growth_factor
                self.num_unskipped = 0
                log.info(f"Increasing scale factor to {self.scale_factor:.3f}")

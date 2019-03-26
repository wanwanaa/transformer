# optim wrapper that implements rate
import numpy as np


class Optim():
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.warmup_steps = config.warmup_steps
        self.model_size = config.model_size
        self.steps = 0
        self.init_lr = np.power(self.model_size, -0.5)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def updata_lr(self):
        self.steps += 1
        lr_scale = np.min([
            np.power(self.steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.steps])
        lr = self.init_lr * lr_scale

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def updata(self):
        self.updata_lr()
        self.optimizer.step()
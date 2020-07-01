import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):

    def __init__(self, residual, res_args, init_params):
        super(BaseOptimizer, self).__init__()
        self.residual = residual  # result_nx1 = self.residual(self.params, *self.res_args)
        self.res_args: list = res_args
        self.params = np.array(init_params).astype(np.float64)
        if self.params.ndim == 1:
            self.params = np.expand_dims(self.params, axis=0).T
        assert self.params.ndim == 2
        assert self.params.shape[1] == 1

    @abstractmethod
    def solve(self):
        pass

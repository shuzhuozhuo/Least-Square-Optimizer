import numpy as np
from abc import ABC, abstractmethod


class BaseResidual(ABC):

    def __init__(self):
        super(BaseResidual, self).__init__()

    @abstractmethod
    def residual_function(self, p, *args):
        pass

    @abstractmethod
    def build_data(self):
        pass

    @abstractmethod
    def jacobi(self):
        pass

    @abstractmethod
    def hessian(self):
        pass

    def forward(self, p, *args):
        p = np.array(p)
        if p.ndim == 1:
            p = np.expand_dims(p, 0).T
        return self.residual_function(p, *args)

    def __call__(self, p, *args):

        return self.forward(p, *args)




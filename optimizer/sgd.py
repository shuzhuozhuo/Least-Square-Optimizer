from .optimizer import BaseOptimizer
import numpy as np
from residual.base_residual import BaseResidual
from numdiff.numdiff import DerivativeEstimator


class SGD(BaseOptimizer):

    def __init__(self, residual, res_args, init_params, momentum=0.0, weight_decay=0.0,
                 lr=0.01, delta_grad=0.001, max_iter=1000, use_num_diff=False, diff_method="3-point"):
        assert isinstance(residual, BaseResidual)
        super(SGD, self).__init__(residual=residual, res_args=res_args, init_params=init_params,
                                  use_num_diff=use_num_diff, diff_method=diff_method)

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr
        self.delta_grad = delta_grad
        self.max_iter = max_iter
        self.gt = np.zeros_like(self.params)

    def calc_delta_p(self):
        """
        loss = 0.5 * (pre_y - gt_y) ** 2
        d(loss) / d(pre_y) = pre_y - gt_y
        d(pre_y) / d(x) = jacobi_matrix
        delta_p = = d(loss) / d(x) = (pre_y - gt_y).T * jacobi_matrix
        :return:
        """

        res, pre_y, gt_y = self.residual(self.params, *self.res_args)
        jacobi_matrix_mxn = self.residual.jacobi(self.params, *self.res_args)
        if jacobi_matrix_mxn is None or self.use_num_diff:
            jacobi_matrix_mxn = self.diff_estimator(self.params)
        diff_y = pre_y - gt_y
        grad_1xn = np.matmul(diff_y.T, jacobi_matrix_mxn)
        grad_nx1 = grad_1xn.T
        self.gt = self.momentum * self.gt + (1 - self.momentum) * grad_nx1

        return self.gt

    def update_lr(self, i):
        # self.lr *= 1 - i / self.max_iter
        pass

    def show_stats(self, delta_p, i):
        residual, _, _ = self.residual(self.params, *self.res_args)
        print("Current grad is: ", np.linalg.norm(delta_p), "Stop at %d Iteration" % i)
        # print("residual", residual)
        print("loss: ", np.mean(np.abs(residual)))

    def solve(self):

        for i in range(self.max_iter):
            delta_p = self.calc_delta_p()
            self.update_lr(i)
            self.params -= self.lr * delta_p
            if np.linalg.norm(delta_p) < self.delta_grad:
                self.show_stats(delta_p, i)
                return self.params
        self.show_stats(delta_p, i)
        return self.params

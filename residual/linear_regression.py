from .base_residual import BaseResidual
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(BaseResidual):

    def __init__(self, weights, points_num, num_range, noise_range=1.0):
        super(LinearRegression, self).__init__()

        assert np.array(weights).ndim == 1
        self.weights = weights

        self.points_num = points_num

        assert isinstance(num_range, (list, tuple, np.ndarray))
        assert len(num_range) == 2
        self.num_range = num_range

        self.noise_range = noise_range

    def build_data(self, is_debug=False):
        """
        line function:
            y = w1*x1 + w2*x2 + w3*x3 + ... + w_n-1 * x_n-1 + wn * xn
        :return: x, y
        """

        length = len(self.weights)
        x_mxn = np.random.random((self.points_num, length)) * (self.num_range[1] - self.num_range[0]) + self.num_range[0]
        noise = np.random.random((self.points_num, 1)) * 2 * self.noise_range - self.noise_range
        x_mxn[:, -1] = 1.0
        y_1xm = (x_mxn * self.weights).sum(axis=1)
        y_mx1 = np.expand_dims(y_1xm, axis=0).T
        y_mx1 += noise
        if is_debug:
            plt.scatter(x_mxn[:, 0], y_1xm)
            plt.show()
        return x_mxn, y_mx1

    def residual_function(self, p_nx1, x_mxn, y_mx1):
        predict_y_m = (x_mxn * p_nx1[:, 0]).sum(axis=1)
        predict_y_mx1 = np.expand_dims(predict_y_m, axis=0).T
        residual = predict_y_mx1 - y_mx1
        return residual, predict_y_mx1, y_mx1

    def jacobi(self, p_nx1, x_mxn, y_mx1):
        """
        jacobi matrix between residual and x
        :return: jacobi_matrix
        """
        return x_mxn

    def hessian(self):
        return None


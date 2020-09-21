from .base_residual import BaseResidual
import numpy as np
import matplotlib.pyplot as plt


class FindCenter(BaseResidual):

    def __init__(self):
        super(FindCenter, self).__init__()
        self.points = None

    def build_data(self, points, is_debug=False):
        """
        line function:
            y = w1*x1 + w2*x2 + w3*x3 + ... + w_n-1 * x_n-1 + wn * xn
        :return: x, y
        """

        self.points = points
        assert points.shape[1] == 2
        points_num = points.shape[0]

        data_nx4 = np.empty((points_num - 1, 4), dtype=np.float32)

        for i in range(points_num - 1):
            data_nx4[i, :2] = self.points[i]
            data_nx4[i, 2:] = self.points[i + 1]

        if is_debug:
            plt.scatter(data_nx4[:, 0], data_nx4[:, 1], cmap="blue")
            # plt.scatter(self.center[0], self.center[1], cmap="red")
            plt.show()
        return data_nx4

    def residual_function(self, p_nx1, data_nx4):

        dist_1 = np.linalg.norm(data_nx4[:, :2] - p_nx1[:, 0], axis=1)
        dist_2 = np.linalg.norm(data_nx4[:, 2:] - p_nx1[:, 0], axis=1)
        print("data_nx4", data_nx4)
        print("dist_1", dist_1)
        print("dist_2", dist_2)
        dist_1_mx1 = np.expand_dims(dist_1, axis=0).T
        dist_2_mx1 = np.expand_dims(dist_2, axis=0).T

        residual = np.abs(dist_1_mx1 - dist_2_mx1)
        print("residual", residual)
        return residual, dist_1_mx1, dist_2_mx1

    def jacobi(self, p_nx1, dist_1_mx1):
        """
        jacobi matrix between residual and x
        :return: jacobi_matrix
        """
        return None

    def hessian(self):
        return None


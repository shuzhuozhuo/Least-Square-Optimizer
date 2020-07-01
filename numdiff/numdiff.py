import numpy as np

EPS = np.finfo(np.float64).eps
RELATIVE_STEP = {"2-point": EPS**0.5,
                 "3-point": EPS**(1/3),
                 "cs": EPS**0.5}


class DerivativeEstimator(object):
    """
    Refer from Scipy, But this class only implement the dense jacobi matrix calculation
    """

    def __init__(self, func, args, rel_step=None, method="3-point", use_one_side=False):
        super(DerivativeEstimator, self).__init__()
        self.func = func
        self.args = args
        self.rel_step = rel_step
        self.method = method
        self.use_one_side = use_one_side

    @staticmethod
    def check_x_shape(x):
        not_equal_to_zero = 0
        x = np.array(x)
        for i in x.shape:
            if i != 1:
                not_equal_to_zero += 1
        if not_equal_to_zero > 1:
            raise ValueError
        return True

    def reshape_array(self, x: np.ndarray):
        reshape_x = None
        if self.check_x_shape(x):
            reshape_x = x.flatten()
        return reshape_x

    @staticmethod
    def get_residual(result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def three_points_method(self, x):
        x0 = self.reshape_array(x.copy())
        length = len(x0)
        h = RELATIVE_STEP["3-point"] if self.rel_step is None else self.rel_step
        h_mat = np.eye(length) * h
        jacobi = []
        for i in range(length):
            if self.use_one_side:
                x1 = x0
                x2 = x0 + 2 * h_mat[i]
                y1 = self.get_residual(self.func(x1, *self.args))
                y2 = self.get_residual(self.func(x2, *self.args))
            else:
                x1 = x0 - h_mat[i]
                x2 = x0 + h_mat[i]
                y1 = self.get_residual(self.func(x1, *self.args))
                y2 = self.get_residual(self.func(x2, *self.args))
            y1 = self.reshape_array(y1)
            y2 = self.reshape_array(y2)
            df = y2 - y1
            dx = x2[i] - x1[i]
            jacobi.append(df / dx)
        jacobi_mxn = np.array(jacobi).T
        return jacobi_mxn

    def two_points_method(self, x):
        x0 = self.reshape_array(x.copy())
        length = len(x0)
        h = RELATIVE_STEP["2-point"] if self.rel_step is None else self.rel_step
        h_mat = np.eye(length) * h
        jacobi = []
        for i in range(length):
            x1 = x0
            x2 = x0 + h_mat[i]
            y1, _, _ = self.func(x1, *self.args)
            y2, _, _ = self.func(x2, *self.args)
            y1 = self.reshape_array(y1)
            y2 = self.reshape_array(y2)
            df = y2 - y1
            dx = x2[i] - x1[i]
            jacobi.append(df / dx)
        jacobi_mxn = np.array(jacobi).T
        return jacobi_mxn

    def calculate_derivative(self, x):
        if self.method == "3-point":
            return self.three_points_method(x=x)
        elif self.method == "2-point":
            return self.two_points_method(x=x)
        else:
            raise(NotImplementedError, "This method is not implemented, Please Use 2-point or 3-point")

    def __call__(self, x):
        return self.calculate_derivative(x=x)


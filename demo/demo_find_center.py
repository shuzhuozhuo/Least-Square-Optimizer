from residual.find_center import FindCenter
from optimizer.sgd import SGD
from optimizer.gaussain_newton import GaussianNewton
import numpy as np

weights = [1, 5, 2, 9, 2, 1]
points_num = 50
init_params = [10, 20]
linear = FindCenter()

points = np.array([[10, 20], [30, 50], [15, 27], [23, 35], [50, 35]])

data_nx4 = linear.build_data(points, is_debug=False)

# sgd = SGD(residual=linear, res_args=[data_nx4], init_params=init_params, lr=0.005, momentum=0.9,
#           max_iter=1000, use_num_diff=True, diff_method="3-point")
gaussian_newton = GaussianNewton(residual=linear, res_args=[data_nx4], init_params=init_params, lr=0.01, momentum=0.9,
                                 max_iter=10000, use_num_diff=True, diff_method="3-point")
# opt_params = sgd.solve()
# print(opt_params)
gopt_params = gaussian_newton.solve()
print(gopt_params)

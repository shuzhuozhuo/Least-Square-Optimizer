from residual.center_regression import CenterRegression
from optimizer.sgd import SGD
from optimizer.gaussain_newton import GaussianNewton

weights = [1, 5]
points_num = 50
init_params = [40, 0]
linear = CenterRegression(center=weights, points_num=points_num, num_range=10, noise_range=1.0)
x, y = linear.build_data(is_debug=True)

sgd = SGD(residual=linear, res_args=[x, y], init_params=init_params, lr=0.005, momentum=0.9,
          max_iter=1000, use_num_diff=True, diff_method="3-point")
gaussian_newton = GaussianNewton(residual=linear, res_args=[x, y], init_params=init_params, lr=1, momentum=0.9,
                                 max_iter=1000, use_num_diff=True, diff_method="3-point")
opt_params = sgd.solve()
print(opt_params)
gopt_params = gaussian_newton.solve()
print(gopt_params)

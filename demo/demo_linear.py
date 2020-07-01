from residual.linear_regression import LinearRegression
from optimizer.sgd import SGD
from optimizer.gaussain_newton import GaussianNewton

weights = [1,5,2, 9, 2, 1]
points_num = 50
init_params = [0, 0,0, 0, 0, 0]
linear = LinearRegression(weights=weights, points_num=points_num, num_range=[0, 10], noise_range=1.0)
x, y = linear.build_data(is_debug=True)

sgd = SGD(residual=linear, res_args=[x, y], init_params=init_params, lr=0.001, momentum=0.9, max_iter=1000)
gaussian_newton = GaussianNewton(residual=linear, res_args=[x, y], init_params=init_params, lr=0.01, momentum=0.9, max_iter=1000)
opt_params = sgd.solve()
print(opt_params)
gopt_params = gaussian_newton.solve()
print(gopt_params)

from scipy.optimize import leastsq, least_squares
import numpy as np
import matplotlib.pyplot as plt


def build_dataset(low, up, a, b, c, num):
    x = np.random.random(num) * (up - low) + low
    x = np.sort(x)
    y = (c + a * x) / (-b)

    noise_x = np.random.random(num) * 0.01
    noise_y = np.random.random(num) * 0.01

    x_noise = x + noise_x
    y_noise = y + noise_y
    # print(x == x_noise)
    # plt.scatter(x_noise, y_noise)
    # plt.show()

    return x_noise, y_noise


def get_dloos_dy(y, ypi):
    return -1 * (y - ypi)

def get_jac(x_nosie):
    x = np.expand_dims(x_nosie, 0).T
    z = np.ones_like(x).astype(np.float32)
    jac = np.hstack([x, z])
    return jac


def get_lr(epoch, lr, total):
    if  epoch < total * 0.8:
        return lr
    elif epoch < total * 0.9:
        return lr / 10
    else:
        return lr / 100


def main():
    low, up = 5, 10
    a, b, c = 8, 4, 5
    num = 50
    x_noise, y_noise = build_dataset(low, up, a, b, c, num)
    jac = get_jac(x_noise)
    lr = 0.1 /50
    sabc = np.array([10000, 10000])
    params = np.expand_dims(sabc, 0).T
    y = np.expand_dims(y_noise, 0).T
    ypi = np.matmul(jac, params)
    # print(ypi.shape)
    mt = np.array([[0], [0]])
    epoch = 10000
    for i in range(epoch):
        ypi = np.matmul(jac, params)

        j = get_dloos_dy(y, ypi)
        grad = np.matmul(j.T, jac)
        t_lr = get_lr(i, lr, epoch)
        # mt = mt * 0.9 + grad.T * 0.1
        mt = grad.T
        params = params - t_lr * mt
        print("ypi", j.sum(), "grad:", mt.T, "params:", params.T)
        # print("grad", lr * grad.T)
        # print(params)
    print(params)
    # print(jac)



if __name__ == '__main__':
    main()


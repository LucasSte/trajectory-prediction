import pickle
import typing
import auto_ks
import matplotlib.pyplot as plt
import numpy as np


class SSLData(typing.NamedTuple):
    dim_1: typing.Any
    dim_2: typing.Any
    mask: typing.Any


class KalmanSmoother:
    h = 1. / 100.0
    lam = 1e-10
    alpha = 1e-4
    n = 4
    m = 2
    A = np.array([[1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    W_neg_sqrt = np.eye(n)  # Process noise
    V_neg_sqrt = np.eye(m)  # Sensor noise
    params_initial = auto_ks.KalmanSmootherParameters(A, W_neg_sqrt, C, V_neg_sqrt)
    params = None

    def __init__(self, process_noise_factor=1, sensor_noise_factor=1):
        self.W_neg_sqrt = process_noise_factor*self.W_neg_sqrt
        self.V_neg_sqrt = sensor_noise_factor*self.V_neg_sqrt

    def prox(self, params, t):
        # Code from https://github.com/cvxgrp/auto_ks
        r = 0.0
        w_neg_sqrt = params.W_neg_sqrt / (t * self.alpha + 1.0)
        idx = np.arrange(w_neg_sqrt.shape[0])
        w_neg_sqrt[idx, idx] = 0.0
        r += self.alpha * np.sum(np.square(w_neg_sqrt))
        w_neg_sqrt[idx, idx] = np.diag(params.W_neg_sqrt)

        v_neg_sqrt = params.V_neg_sqrt / (t * self.alpha + 1.0)
        idx = np.arange(v_neg_sqrt.shape[0])
        v_neg_sqrt[idx, idx] = 0.0
        r += self.alpha * np.sum(np.square(v_neg_sqrt))
        v_neg_sqrt[idx, idx] = np.diag(params.V_neg_sqrt)

        return auto_ks.KalmanSmootherParameters(self.A, w_neg_sqrt, self.C, v_neg_sqrt), r

    def fit_for_series(self, x, y, mask, lr, niter):
        y = np.stack([x, y], axis=1)
        K_pos = np.array(mask, dtype=np.bool)
        K = np.repeat(K_pos[:, None], 2, axis=1)
        self.params, _ = auto_ks.tune(self.params_initial, self.prox, y, K, self.lam, lr=lr, verbose=True, niter=niter)

    def smooth(self, x, y, mask):
        y = np.stack([x, y], axis=1)
        K_pos = np.array(mask, dtype=np.bool)
        K = np.repeat(K_pos[:, None], 2, axis=1)

        return auto_ks.kalman_smoother(self.params, y, K, self.lam)

    def test_for_series(self, x, y, mask):
        xhat, yhat, DT = self.smooth(x, y, mask)

        plt.plot()
        plt.plot(x, y, 'r')
        plt.plot(xhat[:, 0], xhat[:, 2], 'b')
        plt.legend(['original', 'smoothed'])
        plt.show()

    def save_params(self, file_name):
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name):
        f = open(file_name, 'rb')
        self.params = pickle.load(f)

    def call(self, training_data: SSLData, testing_data: SSLData, file_name, lr=1e-2, niter=25):
        self.fit_for_series(training_data.dim_1, training_data.dim_2, training_data.mask, lr, niter)
        self.test_for_series(testing_data.dim_1, testing_data.dim_2, testing_data.mask)
        self.save_params(file_name)
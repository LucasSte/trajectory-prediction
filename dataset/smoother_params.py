import pickle
import matplotlib.pyplot as plt
from kalman_smoother import SSLData, KalmanSmoother
import numpy as np


def get_robots():
    f = open('data_set_1.pkl', 'rb')
    robots = pickle.load(f)
    robot_1 = robots['blue'][0][1]
    robot_2 = robots['yellow'][0][2]
    return robot_1, robot_2


def get_robot_position_series_params():
    robot_1, robot_2 = get_robots()

    training_data = SSLData(
        dim_1=robot_1['x'],
        dim_2=robot_1['y'],
        mask=robot_1['mask']
    )

    testing_data = SSLData(
        dim_1=robot_2['x'],
        dim_2=robot_2['y'],
        mask=robot_2['mask']
    )

    fitter = KalmanSmoother(1, 0.1)
    fitter.call(training_data, testing_data, 'position_series_params', lr=1e-2, niter=25)


def get_robot_heading_series_params():
    robot_2, robot_1 = get_robots()
    psi_1 = np.array(robot_1['psi'])
    sin_1 = np.sin(psi_1)
    cos_1 = np.cos(psi_1)

    psi_2 = np.array(robot_2['psi'])
    sin_2 = np.sin(psi_2)
    cos_2 = np.cos(psi_2)

    fitter = KalmanSmoother(1, 0.1)
    fitter.fit_for_series(sin_1, cos_1, robot_1['mask'], lr=1e-4, niter=25)
    xhat, _, _ = fitter.smooth(sin_2, cos_2, robot_2['mask'])
    psi_2_smoothed = np.arctan2(xhat[:, 0], xhat[:, 2])

    time = np.arange(0, np.shape(psi_2)[0])
    plt.figure()
    plt.plot(time, psi_2, 'b')
    plt.plot(time, psi_2_smoothed, 'r')
    plt.legend(['original', 'smoothed'])
    plt.show()

    fitter.save_params('heading_series_params')


def get_ball_position_series_params():
    f = open('data_set_1.pkl', 'rb')
    file_data = pickle.load(f)
    ball = file_data['ball']
    ball_1 = ball[1]
    ball_2 = ball[0]

    training_data = SSLData(
        dim_1=ball_1['x'],
        dim_2=ball_1['y'],
        mask=ball_1['mask']
    )

    testing_data = SSLData(
        dim_1=ball_2['x'],
        dim_2=ball_2['y'],
        mask=ball_2['mask']
    )

    fitter = KalmanSmoother(10000, 1000)
    fitter.call(training_data, testing_data, 'ball_positions_series_params', lr=1e-2, niter=25)

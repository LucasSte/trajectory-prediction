import pickle

from kalman_smoother import KalmanSmoother
import numpy as np


class Smoother:
    position_smoother = KalmanSmoother()
    psi_smoother = KalmanSmoother()
    ball_smoother = KalmanSmoother()

    def __init__(self):
        self.position_smoother.load_params('position_series_params')
        self.psi_smoother.load_params('heading_series_params')
        self.ball_smoother.load_params('ball_positions_series_params')

    def smooth_psi(self, psi, mask):
        psi_sin = np.sin(psi)
        psi_cos = np.cos(psi)

        xhat, _, _ = self.psi_smoother.smooth(psi_sin, psi_cos, mask)
        return np.arctan2(xhat[:, 0], xhat[:, 2])

    def process_robots_data(self, robots, stop_id, response):
        for k in range(len(robots)):
            for robot_id, data in robots[k].items():
                if len(data['x']) > 101:
                    xhat, _, _ = self.position_smoother.smooth(data['x'], data['y'], data['mask'])
                    response['position']['x'].append(xhat[:, 0])
                    response['position']['y'].append(xhat[:, 2])

                    response['speed']['x'].append(xhat[:, 1])
                    response['speed']['y'].append(xhat[:, 3])

                    new_psi = self.smooth_psi(np.array(data['psi']), data['mask'])
                    response['psi'].append(new_psi)

                    response['time_c'].append(data['time_c'])
                    response['stop_id'].append(stop_id[k])


    def process_ball_data(self, ball, stop_id, response):
        for k in range(len(ball)):
            cur = ball[k]
            xhat, _, _ = self.ball_smoother.smooth(cur['x'], cur['y'], cur['mask'])
            response[stop_id[k]] = {
                'x': xhat[:, 0],
                'y': xhat[:, 2],
                'v_x': xhat[:, 1],
                'v_y': xhat[:, 2],
                'time_c': cur['time_c']
            }

    def smooth_data(self, source_file, dest_file):
        robots_b = {'position': {'x': [], 'y': []}, 'speed': {'x': [], 'y': []}, 'psi': [], 'stop_id': [], 'time_c': []}
        robots_y = {'position': {'x': [], 'y': []}, 'speed': {'x': [], 'y': []}, 'psi': [], 'stop_id': [], 'time_c': []}
        ball = {}

        file = open(source_file + '.pkl', 'rb')
        data = pickle.load(file)
        self.process_robots_data(data['blue'], data['stop_id'], robots_b)
        self.process_robots_data(data['yellow'], data['stop_id'], robots_y)
        self.process_ball_data(data['ball'], data['stop_id'], ball)

        all_data = {
            'yellow': robots_y,
            'blue': robots_b,
            'ball': ball,
        }

        with open(dest_file + '.pkl', 'wb') as f:
            pickle.dump(all_data, f)


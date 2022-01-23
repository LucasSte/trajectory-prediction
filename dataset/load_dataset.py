import bisect
import pickle
import numpy as np
import itertools


def load_data(file_path):
    f = open(file_path + '.pkl', 'rb')
    robots = pickle.load(f)
    blue = robots['blue']
    yellow = robots['yellow']
    ball = robots['ball']

    return blue, yellow, ball


def merge_teams(team_1, team_2):
    data = dict()
    data['position'] = {}
    data['speed'] = {}

    data['position']['x'] = team_1['position']['x'] + team_2['position']['x']
    data['position']['y'] = team_1['position']['y'] + team_2['position']['y']
    data['speed']['x'] = team_1['speed']['x'] + team_2['speed']['y']
    data['speed']['y'] = team_1['speed']['x'] + team_2['speed']['y']
    data['psi'] = team_1['psi'] + team_2['psi']
    data['stop_id'] = team_1['stop_id'] + team_2['stop_id']
    data['time_c'] = team_1['time_c'] + team_2['time_c']

    return data


def get_avg_std_for_robots(data):
    flat_x_coord = np.array(list(itertools.chain(*data['position']['x'])))
    flat_y_coord = np.array(list(itertools.chain(*data['position']['y'])))
    flat_vx = np.array(list(itertools.chain(*data['speed']['x'])))
    flat_vy = np.array(list(itertools.chain(*data['speed']['y'])))

    flat_psi = np.array(list(itertools.chain(*data['psi'])))

    x_coord_avg, x_coord_std = np.mean(flat_x_coord), np.std(flat_x_coord)
    vx_avg, vx_std = np.mean(flat_vx), np.std(flat_vx)
    y_coord_avg, y_coord_std = np.mean(flat_y_coord), np.std(flat_y_coord)
    vy_avg, vy_std = np.mean(flat_vy), np.std(flat_vy)

    psi_avg, psi_std = np.mean(flat_psi), np.std(flat_psi)

    avg = np.asarray((x_coord_avg, y_coord_avg, vx_avg, vy_avg, psi_avg))
    std = np.asarray((x_coord_std, y_coord_std, vx_std, vy_std, psi_std))

    return avg, std


class LoadDataSet:

    def __init__(self, look_back, look_forth):
        self.look_back = look_back
        self.look_forth = look_forth

    def get_ball_data(self, ball, stop_id, time_c):
        local = ball[stop_id]
        end = bisect.bisect_left(local['time_c'], time_c, 0, len(local['time_c']))
        start = max(end - self.look_back, 0)
        size = end - start

        pos_x = local['x'][start:end]
        pos_y = local['y'][start:end]
        speed_x = local['v_x'][start:end]
        speed_y = local['v_y'][start:end]
        mask = [True]*self.look_back

        if size < self.look_back:
            diff = self.look_back - size
            pos_x = np.concatenate([np.array([local['x'][0]]*diff), pos_x], axis=0)
            pos_y = np.concatenate([np.array([local['y'][0]*diff]), pos_y], axis=0)
            speed_x = np.concatenate([np.array([local['v_x'][0]*diff]), speed_x], axis=0)
            speed_y = np.concatenate([np.array([local['v_y'][0]*diff]), speed_y], axis=0)
            mask[0:diff] = [False]*diff

        return np.stack([pos_x, pos_y, speed_x, speed_y]).T, np.array(mask, dtype=np.bool)
    
    @staticmethod
    def get_robot_data(data, index):
        x = data['pos']['x'][index]
        y = data['pos']['y'][index]
        v_x = data['speed']['x'][index]
        v_y = data['speed']['y'][index]
        psi = data['psi'][index]

        return np.stack([x, y, v_x, v_y, psi]).T

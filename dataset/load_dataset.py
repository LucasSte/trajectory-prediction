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

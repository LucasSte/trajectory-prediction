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


def get_avg_std_for_robots_merged(data):
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


def get_avg_std_for_robots(robots_data):
    merged_data = None
    for item in robots_data:
        if merged_data is None:
            merged_data = item
        else:
            merged_data = merge_teams(merged_data, item)

    return get_avg_std_for_robots_merged(merged_data)


def get_avg_std_for_ball(ball_data):
    ball_x = []
    ball_y = []
    ball_vx = []
    ball_vy = []

    for ball_data_set in ball_data:
        for _, value in ball_data_set.items():
            ball_x.append(value['x'])
            ball_y.append(value['y'])
            ball_vx.append(value['v_x'])
            ball_vy.append(value['v_y'])

    ball_x = np.concatenate(ball_x, axis=0)
    ball_y = np.concatenate(ball_y, axis=0)
    ball_vx = np.concatenate(ball_vx, axis=0)
    ball_vy = np.concatenate(ball_vy, axis=0)

    x_ball_avg, x_ball_std = np.mean(ball_x), np.std(ball_x)
    y_ball_avg, y_ball_std = np.mean(ball_y), np.std(ball_y)
    vx_ball_avg, vx_ball_std = np.mean(ball_vx), np.std(ball_vx)
    vy_ball_avg, vy_ball_std = np.mean(ball_vy), np.std(ball_vy)

    ball_avg = np.asarray((x_ball_avg, y_ball_avg, vx_ball_avg, vy_ball_avg))
    ball_std = np.asarray((x_ball_std, y_ball_std, vx_ball_std, vy_ball_std))

    return ball_avg, ball_std


class LoadDataSet:

    ball_avg = None
    ball_std = None
    robots_avg = None
    robots_std = None

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

    '''
    Receives a list of .pkl files to load
    '''
    def load_data_as_train(self, data_sets: list):
        robot_data = []
        ball_data = []
        for elem in data_sets:
            blue, yellow, ball = load_data(elem)
            new_data = merge_teams(blue, yellow)
            robot_data.append(new_data)
            ball_data.append(ball)

        self.ball_avg, self.ball_std = get_avg_std_for_ball(ball_data)
        self.robots_avg, self.robots_std = get_avg_std_for_robots(robot_data)

        # TODO: Prepare single trajectories



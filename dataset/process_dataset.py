from read_logs import process_log
from smooth_series import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params

data_set_files = ['data_set_1', 'data_set_2', 'data_set_3']

for file in data_set_files:
    process_log(file)

get_ball_position_series_params()
get_robot_position_series_params()
get_robot_heading_series_params()

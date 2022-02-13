from read_logs import process_log
from smoother_params import get_ball_position_series_params, get_robot_position_series_params, get_robot_heading_series_params
from smooth_data import Smoother

data_set_files = ['data_set_1', 'data_set_2', 'data_set_3']
processed_data_files = ['proc_set_1', 'proc_set_2', 'proc_set_3']

for file in data_set_files:
    print(file)
    process_log(file)

print('---- Processing ball data ----')
get_ball_position_series_params()
print('---- Processing robot position data ----')
get_robot_position_series_params()
print('---- Processing robot heading data ----')
get_robot_heading_series_params()

smoother = Smoother()
for (source, dest) in zip(data_set_files, processed_data_files):
    smoother.smooth_data(source, dest)


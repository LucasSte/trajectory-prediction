import math
from reader.Reader import Reader
from reader.MessageType import MessageType
import pickle


def add_robot_measurement(dic, elem, pkt):
    dic[elem.robot_id]['x'].append(elem.x)
    dic[elem.robot_id]['y'].append(elem.y)
    dic[elem.robot_id]['psi'].append(elem.orientation)
    dic[elem.robot_id]['time_c'].append(pkt.detection.t_capture)
    dic[elem.robot_id]['mask'].append(True)


def add_robot_element(dic, st, elem, pkt):
    # Check if element is in dictionary
    if elem.robot_id not in dic:
        dic[elem.robot_id] = {}
        dic[elem.robot_id]['x'] = []
        dic[elem.robot_id]['y'] = []
        dic[elem.robot_id]['psi'] = []
        dic[elem.robot_id]['time_c'] = []
        dic[elem.robot_id]['mask'] = []

    # If the robot has not been processed yet
    if elem.robot_id not in st:
        diff = - 50
        # Get different between current and last processed measurement
        if len(dic[elem.robot_id]['time_c']) > 0:
            diff = pkt.detection.t_capture - dic[elem.robot_id]['time_c'][-1]

        if diff == -50 or (0.01 < diff < 0.022):
            add_robot_measurement(dic, elem, pkt)
            st.add(elem.robot_id)
        # If the difference is greater than 0.022, we try to interpolate measurements, by setting
        # their mask as false. It means they will be calculated later using a Kalman smoother
        elif diff >= 0.022:
            steps = math.floor(diff * 60)
            for k in range(steps):
                if (pkt.detection.t_capture - dic[elem.robot_id]['time_c'][-1] - (1 / 60)) < 0.01:
                    break
                # There is no problem in repeating the last measurement, as we are going to calculate
                # these values later.
                dic[elem.robot_id]['x'].append(dic[elem.robot_id]['x'][-1])
                dic[elem.robot_id]['y'].append(dic[elem.robot_id]['y'][-1])
                dic[elem.robot_id]['psi'].append(dic[elem.robot_id]['psi'][-1])
                dic[elem.robot_id]['time_c'].append(dic[elem.robot_id]['time_c'][-1] + (1 / 60))
                dic[elem.robot_id]['mask'].append(False)
            add_robot_measurement(dic, elem, pkt)
            st.add(elem.robot_id)


def add_ball_measurement(dic, elem, pkt):
    dic['x'].append(elem.x)
    dic['y'].append(elem.y)
    dic['mask'].append(True)
    dic['time_c'].append(pkt.detection.t_capture)


def add_ball_element(dic, elem, pkt):
    diff = -50
    if len(dic['time_c']) > 0:
        diff = pkt.detection.t_capture - dic['time_c'][-1]

    if diff == -50 or (0.01 < diff < 0.022):
        add_ball_measurement(dic, elem, pkt)
    # The same logic for interpolation here
    elif diff >= 0.022:
        steps = math.floor(diff * 60)
        for k in range(steps):
            if (pkt.detection.t_capture - dic['time_c'][-1] - (1 / 60)) < 0.01:
                break
            dic['x'].append(dic['x'][-1])
            dic['y'].append(dic['y'][-1])
            dic['time_c'].append(dic['time_c'][-1] + (1 / 60))
            dic['mask'].append(False)
        add_ball_measurement(dic, elem, pkt)


def process_log(path):
    reader = Reader(path + '.log')
    reader.read_header()
    collect = False
    i = 0

    robots_b = {}
    robots_y = {}
    ball = {'x': [], 'y': [], 'time_c': [], 'mask': []}

    all_data = {'yellow': [], 'blue': [], 'ball': [], 'stop_id': []}

    while reader.has_next():
        msg_type = reader.decode_msg()
        if msg_type == MessageType.MESSAGE_SSL_VISION_2010 or msg_type == MessageType.MESSAGE_SSL_VISION_2014:
            wrapper_packet = reader.get_wrapper_packet()
            # We discard the first packet to decrease noise
            if wrapper_packet.detection.t_capture == 0:
                continue

            if len(wrapper_packet.detection.robots_blue) > 0 and collect:
                st = set()
                for elem in wrapper_packet.detection.robots_blue:
                    add_robot_element(robots_b, st, elem, wrapper_packet)

            if len(wrapper_packet.detection.robots_yellow) > 0 and collect:
                st = set()
                for elem in wrapper_packet.detection.robots_yellow:
                    add_robot_element(robots_y, st, elem, wrapper_packet)

            if len(wrapper_packet.detection.balls) > 0 and collect:
                elem = wrapper_packet.detection.balls[0]
                add_ball_element(ball, elem, wrapper_packet)

        elif msg_type == MessageType.MESSAGE_SSL_REFBOX_2013:
            referee_packet = reader.get_referee_packet()
            command = referee_packet.command

            # Command that start the game (NORMAL_START = 2 OR FORCE_START=3)
            if command == 2 or command == 3:
                collect = True
            # Every other command stops the game
            elif collect:
                collect = False
                all_data['blue'].append(robots_b)
                all_data['yellow'].append(robots_y)
                all_data['ball'].append(ball)
                all_data['stop_id'].append(i)
                i += 1
                robots_b = {}
                robots_y = {}
                ball = {'x': [], 'y': [], 'z': [], 'time_c': [], 'mask': []}
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(all_data, f)
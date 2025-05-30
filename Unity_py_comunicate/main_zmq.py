# main_zmq.py
import threading
import time
import numpy as np
import zmq
import json
from UR5_admittance.cc_admittance_2022 import Admittance
from zmq_modules import zmq_unity_pull_target, zmq_unity_push_cursor, zmq_pub_state_detect
from state_detect import state_detect_worker
import multiprocessing
import pandas as pd
import tobii_research as tr
import sys
from URTracker_mcts import MCTSControlThread


mcts_controller = None 

def pos_g2r(r_r, r_g, pos_cg, pos_cr, pos_g):
    k_x = r_r/r_g
    k_y = -r_r/r_g
    b = [pos_cr[0] - k_x*pos_cg[0], pos_cr[1] - k_y*pos_cg[1]]
    return [k_x * pos_g[0] + b[0], k_y * pos_g[1] + b[1]]

def pos_r2g(r_r, r_g, pos_cg, pos_cr, pos_r):
    k_x = r_g/r_r
    k_y = -r_g/r_r
    b = [pos_cg[0] - k_x * pos_cr[0], pos_cg[1] - k_y * pos_cr[1]]
    return [k_x * pos_r[0] + b[0], k_y * pos_r[1] + b[1]]

def gaze_data_callback(gaze_data):
    '''更新并打印最新的gazedata'''
    # Print gaze points of left and right eye
    global latest_gaze_data
    latest_gaze_data = gaze_data

if __name__ == "__main__":
    # 初始化参数
    r_g, r_r = 4.0, 0.1
    pos_cg, pos_cr = [0.0, 0.0], [0.625, -0.050]

    # ZMQ连接
    target_gen = zmq_unity_pull_target.get_target_generator()
    send_cursor = zmq_unity_push_cursor.send_cursor_position
    pub_state = zmq_pub_state_detect.broadcast_state_result

    # 初始化UR5控制器
    g2r_q, r2g_q, cmd_q = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
    ur5 = Admittance()
    threading.Thread(target=ur5.run_test, args=(g2r_q, r2g_q, cmd_q), daemon=True).start()

    # 启动状态识别子进程
    task_q, result_q = multiprocessing.Queue(), multiprocessing.Queue()
    detect_proc = multiprocessing.Process(target=state_detect_worker, args=(task_q, result_q), daemon=True)
    detect_proc.start()
    stage_id = 0
    state_flag = 0
    motor_flag = 0
    e_c = 0
    e_m = 0
    I_FB = 0
    I_ASST = 0

    # 新增 MCTS 控制线程初始化
    context = zmq.Context()
    mcts_pub_socket = context.socket(zmq.PUB)
    mcts_pub_socket.bind("tcp://*:5557")  # 新端口用于向unity发送 MCTS 参数
    # MCTS 线程订阅``
    mcts_sub = context.socket(zmq.SUB)
    mcts_sub.connect("tcp://localhost:6002")
    mcts_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    mcts_thread = MCTSControlThread(mcts_sub, cmd_q, mcts_pub_socket)
    mcts_thread.start()


    use_tobii = True
    use_robot = True
    latest_gaze_data = None
    ''' 启动tobii '''
    if use_tobii:
        try:
            found_eyetrackers = tr.find_all_eyetrackers()
            my_eyetracker = found_eyetrackers[0]
            print("Address: " + my_eyetracker.address)
            print("Model: " + my_eyetracker.model)
            print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
            print("Serial number: " + my_eyetracker.serial_number)
            my_eyetracker.subscribe_to(
                tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True
            )
        except:
            print("No EyeTracker found")
            sys.exit()

    if use_tobii:
            left_gaze = latest_gaze_data['left_gaze_point_on_display_area']
            right_gaze = latest_gaze_data['right_gaze_point_on_display_area']

            # 平均左右眼的注视点
            # TODO: 添加从unity json获取当前屏幕像素的长宽
            gaze_point_x = ((left_gaze[0] + right_gaze[0]) / 2-0.5) * 19.2 if left_gaze[0] > 0 and right_gaze[0] > 0 else None
            gaze_point_y = ((left_gaze[1] + right_gaze[1]) / 2-0.5) * 10.8 if left_gaze[1] > 0 and right_gaze[1] > 0 else None
            pos_gaze = [gaze_point_x, -gaze_point_y]

    # 定义数据 Dataframe
    data_columns = ['time', 'target_x', 'target_y', 'cursor_x', 'cursor_y', 'Hex_x', 'Hex_y', 'Hex_z','force_norm', 'linear_x', 'linear_y', 'linear_z',
                    'pose_x', 'pose_y', 'pose_z', 'pose_rx', 'pose_ry', 'pose_rz','Gaze_x','Gaze_y', 'stage_id', 'congnitive_flag', 'motor_flag', 'I_FB', 'I_ASST']

    df = pd.DataFrame(columns=data_columns)

    # 数据窗口
    df_window = []

    while True:
        try:
            # 1. 接收来自Unity的目标点
            target = next(target_gen)
            target_r = pos_g2r(r_r, r_g, pos_cg, pos_cr, [target["x"], target["y"]])
            g2r_q.put(target_r)

            # 2. 接收来自UR5的末端状态
            if not r2g_q.empty():
                state = r2g_q.get()
                cursor_g = pos_r2g(r_r, r_g, pos_cg, pos_cr, [state["pose_x"], state["pose_y"]])
                send_cursor(*cursor_g)

            # 3. 数据滑窗加入识别窗口
            row = {"time":  time.time(), "target_x": target["x"], "target_y": target["y"],
                "cursor_x": cursor_g[0], "cursor_y": cursor_g[1],
                "Hex_x": state["Hex_x"], "Hex_y": state["Hex_y"],
                "force_norm": state["force_norm"], "linear_x": state["linear_x"],
                "linear_y": state["linear_y"], "linear_z": state["linear_z"],
                "pose_x": state["pose_x"], "pose_y": state["pose_y"],
                "pose_z": state["pose_z"], "pose_rx": state["pose_rx"],
                "pose_ry": state["pose_ry"], "pose_rz": state["pose_rz"],
                "Gaze_x": pos_gaze[0], "Gaze_y": pos_gaze[1],
                "stage_id": state["stage_id"], "state_flag": state_flag,
                "I_FB": state["I_FB"], "I_ASST": state["I_ASST"]
            }
            df_window.append(row)


            if len(df_window) >= 150:  # 滑窗长度
                df = pd.DataFrame(df_window[-150:])
                task_q.put((df, "Detect"))
            
            # 4. 接收识别结果并广播
            if not result_q.empty():
                result = result_q.get()
                if result['status'] == 'success':
                    pub_state(result['data'])  # 广播 state_flag, e_cog_norm, e_mot_norm
                    state_flag = result['data']['state_flag']
                    e_c = result['data']['e_cog_norm']
                    e_m = result['data']['e_mot_norm']

        except KeyboardInterrupt:
            break

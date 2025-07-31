#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' 重构 UNITY 与 python 的通信 '
__author__ = 'Chen Chen'
__date__ = '2025/07/28'
__version__ = '0.1'

import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(main_dir, 'Dynamic_Learning'))
# main_zmq.py
import threading
import time
import numpy as np
import zmq
import json
from UR5_admittance import Admittance_2022
from zmq_modules import zmq_unity_pull_target, zmq_unity_push_cursor, zmq_pub_state_detect
from state_detect import state_detect_worker
import multiprocessing
import pandas as pd
import tobii_research as tr
from URTracker_mcts import MCTSControlThread
import traceback
import queue
import logging


latest_gaze_data = None

def pos_g2r(r_r, r_g, pos_cg, pos_cr, pos_g):
    k_x = r_r/r_g
    k_y = r_r/r_g
    b = [pos_cr[0] - k_x*pos_cg[0], pos_cr[1] - k_y*pos_cg[1]]
    return [k_x * pos_g[0] + b[0], k_y * pos_g[1] + b[1]]

def pos_r2g(r_r, r_g, pos_cg, pos_cr, pos_r):
    k_x = r_g/r_r
    k_y = r_g/r_r
    b = [pos_cg[0] - k_x * pos_cr[0], pos_cg[1] - k_y * pos_cr[1]]
    return [k_x * pos_r[0] + b[0], k_y * pos_r[1] + b[1]]

def gaze_data_callback(gaze_data):
    '''更新并打印最新的gazedata'''
    global latest_gaze_data
    latest_gaze_data = gaze_data
def setup_logger(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


if __name__ == "__main__":
    setup_logger(debug=True)
    # 初始化参数
    r_g, r_r = 4.0, 0.1
    pos_cg, pos_cr = [0.0, 0.0], [0.625, -0.050]
    pos_gaze = [0.0,0.0]
    cursor_g = [0.0,0.0]
    target_g = [0.0,0.0]
    target = {"x":0.0, "y":0.0}

    stage_id = 0
    state_flag = 0
    motor_flag = 0
    e_c = 0
    e_m = 0
    I_FB = 0
    I_ASST = 0

    use_tobii = True
    use_robot = True
    # 定义数据 Dataframe
    data_columns = ['time', 'target_x', 'target_y', 'cursor_x', 'cursor_y', 'Hex_x', 'Hex_y', 'Hex_z','force_norm', 'linear_x', 'linear_y', 'linear_z',
                    'pose_x', 'pose_y', 'pose_z', 'pose_rx', 'pose_ry', 'pose_rz','Gaze_x','Gaze_y', 'stage_id', 'congnitive_flag', 'motor_flag', 'I_FB', 'I_ASST']
    df = pd.DataFrame(columns=data_columns)

    # 数据窗口
    df_window = []

    
    ''' 启动tobii '''
    if use_tobii:
        try:
            found_eyetrackers = tr.find_all_eyetrackers()
            my_eyetracker = found_eyetrackers[0]
            logging.info("Address: " + my_eyetracker.address)
            logging.info("Model: " + my_eyetracker.model)
            logging.info("Name (It's OK if this is empty): " + my_eyetracker.device_name)
            logging.info("Serial number: " + my_eyetracker.serial_number)
            my_eyetracker.subscribe_to(
                tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True
            )
        except:
            logging.warning("No EyeTracker found")
            sys.exit()

    
    # ZMQ连接
    context = zmq.Context()
    # 创建sub套接字，接收来自unity的Target信息
    target_sub_socket = context.socket(zmq.SUB)
    target_sub_socket.setsockopt(zmq.CONFLATE, 1)
    target_sub_socket.setsockopt(zmq.RCVHWM, 1)    # 设置接收高水位为1
    target_sub_socket.bind("tcp://localhost:6005")
    target_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    # 创建pub套接字，向unity发送cursor信息
    cursor_pub_socket = context.socket(zmq.PUB)
    cursor_pub_socket.bind("tcp://*:6001")
    # 添加发送端配置
    cursor_pub_socket.setsockopt(zmq.SNDHWM, 1)  # 设置发送高水位为1
    cursor_pub_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
    
    # 初始化UR5控制器
    g2r_q, r2g_q, cmd_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    ur5 = Admittance_2022()
    ur5_thread = threading.Thread(target=ur5.run_test, args=(g2r_q, r2g_q, cmd_q), daemon=True)
    ur5_thread.start()
    
    # # 启动状态识别子进程
    # task_q, result_q = multiprocessing.Queue(), multiprocessing.Queue()
    # detect_proc = multiprocessing.Process(target=state_detect_worker, args=(task_q, result_q), daemon=True)
    # detect_proc.start()

    # 等待所有设备线程就绪
    time.sleep(2)
    # # 新增 MCTS 控制线程初始化
    
    # mcts_pub_socket = context.socket(zmq.PUB)
    # mcts_pub_socket.bind("tcp://*:5557")  # 新端口用于向unity发送 MCTS 参数
    # # MCTS 线程订阅``
    # mcts_sub = context.socket(zmq.SUB)
    # mcts_sub.connect("tcp://localhost:6002")
    # mcts_sub.setsockopt_string(zmq.SUBSCRIBE, '')

    # mcts_thread = MCTSControlThread(mcts_sub, cmd_q, mcts_pub_socket)
    # mcts_thread.start()

    # 添加循环频率控制
    target_frequency = 50  # 50Hz
    target_period = 1.0 / target_frequency  # 0.02秒
    last_loop_time = time.time()

    while True:
        # 控制循环频率
        current_time = time.time()
        elapsed_time = current_time - last_loop_time
        
        if elapsed_time < target_period:
            # 如果还没到时间，短暂休眠
            sleep_time = target_period - elapsed_time
            time.sleep(sleep_time)
            current_time = time.time()  # 更新时间
        
        last_loop_time = current_time
        # 获取眼动仪数据
        if latest_gaze_data is not None:
            left_gaze = latest_gaze_data['left_gaze_point_on_display_area']
            right_gaze = latest_gaze_data['right_gaze_point_on_display_area']

            # 平均左右眼的注视点
            # TODO: 添加从unity json获取当前屏幕像素的长宽
            gaze_point_x = ((left_gaze[0] + right_gaze[0]) / 2-0.5) * 19.2 if left_gaze[0] > 0 and right_gaze[0] > 0 else None
            gaze_point_y = ((left_gaze[1] + right_gaze[1]) / 2-0.5) * 10.8 if left_gaze[1] > 0 and right_gaze[1] > 0 else None
            if gaze_point_x is not None and gaze_point_y is not None:
                pos_gaze = [gaze_point_x, -gaze_point_y]

        # 接收来自UR5的末端状态
        if not r2g_q.empty():
            state = r2g_q.get()
            cursor_g = pos_r2g(r_r, r_g, pos_cg, pos_cr, [state["pose_x"], state["pose_y"]])
        
        # 接收来自Unity的数据
        ## target
        try:
            target_s = target_sub_socket.recv_string(flags=zmq.NOBLOCK)
            target = json.loads(target_s)
            logging.debug('接收到 target: %f, %f',target["x"], target["y"])
            target_r = pos_g2r(r_r, r_g, pos_cg, pos_cr, [target["x"], target["y"]])
            g2r_q.put(target_r)
        except zmq.Again:
            logging.warning('未接收到target')
            time.sleep(0.001)  # 1ms
        except zmq.ZMQError as e:
            logging.error(f"接收Unity数据时ZMQ错误: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"解析Unity数据JSON失败: {e}")
        except Exception as e:
            logging.error(f"接收Unity数据时未知错误: {e}")
            
            

        # 保存数据
        row = {"time":  time.time(), "target_x": target["x"], "target_y": target["y"],
            "cursor_x": cursor_g[0], "cursor_y": cursor_g[1],
            "Hex_x": state["Hex_x"], "Hex_y": state["Hex_y"],
            "force_norm": state["force_norm"], "linear_x": state["linear_x"],
            "linear_y": state["linear_y"], "linear_z": state["linear_z"],
            "pose_x": state["pose_x"], "pose_y": state["pose_y"],
            "pose_z": state["pose_z"], "pose_rx": state["pose_rx"],
            "pose_ry": state["pose_ry"], "pose_rz": state["pose_rz"],
            "Gaze_x": pos_gaze[0], "Gaze_y": pos_gaze[1],
            "stage_id":stage_id, "state_flag": state_flag,
            "I_FB": I_FB, "I_ASST": I_ASST
        }
        df_window.append(row)

        ''' 输出数据 '''
        # 向Unity发送cursor数据
        cursor_data = {'x':cursor_g[0],'y':cursor_g[1]}
        cursor_pub_socket.send_json(cursor_data)
        logging.debug('发送光标：%f, %f',cursor_g[0], cursor_g[1])
        



        # if len(df_window) >= 150:  # 滑窗长度
        #     df = pd.DataFrame(df_window[-150:])
        #     task_q.put((df, "Detect"))
        
        # # 4. 接收识别结果并广播
        # if not result_q.empty():
        #     result = result_q.get()
        #     if result['status'] == 'success':
        #         pub_state(result['data'])  # 广播 state_flag, e_cog_norm, e_mot_norm
        #         state_flag = result['data']['state_flag']
        #         e_c = result['data']['e_cog_norm']
        #         e_m = result['data']['e_mot_norm']

            
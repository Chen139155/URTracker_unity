"""
Function:
    该代码通过命名管道与线程间共享内存分别与unity和ur5控制器交换数据,并更具指令调用动态模式识别方法
Author:
    CC
"""
import sys
import os
sys.path.append('./UR5_admittance')
sys.path.append(os.path.join(os.path.dirname(__file__), 'Dynamic_Learning'))
import pygame
import math
import pygame_widgets
import random
import threading
import time
import torch
import copy
import pandas as pd
from UR5_admittance.cc_admittance_2022 import Admittance
from Dynamic_Learning import DL_cc_detect as Dcd
from PipeCommunication_0326 import PipeCommunication
import queue
import multiprocessing
import zmq

# pip install tobii_research as tr
import tobii_research as tr

from state_detect import compute_state_flag


latest_gaze_data = None

def gaze_data_callback(gaze_data):
    '''更新并打印最新的gazedata'''
    # Print gaze points of left and right eye
    global latest_gaze_data
    latest_gaze_data = gaze_data

def pos_g2r(r_r, r_g, pos_cg, pos_cr, pos_g):
    '''计算game坐标系到robot坐标的转换'''
    # 坐标转换系数
    R_r = r_r
    R_g = r_g
    pos_cg = pos_cg
    pos_cr = pos_cr
    k_x = R_r/R_g
    k_y = -R_r/R_g
    b = [0.0,0.0]
    b[0] = pos_cr[0] - k_x*pos_cg[0]
    b[1] = pos_cr[1] - k_y*pos_cg[1]
    # 计算转换坐标
    pos_r = [0.0,0.0]
    pos_r[0] = k_x * pos_g[0] + b[0]
    pos_r[1] = k_y * pos_g[1] + b[1]
    return pos_r

def pos_r2g(r_r, r_g, pos_cg, pos_cr, pos_r):
    '''计算robot坐标系到game坐标的转换'''
    # 坐标转换系数
    R_r = r_r
    R_g = r_g
    pos_cg = pos_cg
    pos_cr = pos_cr
    k_x = R_g/R_r
    k_y = -R_g/R_r
    b = [0.0,0.0]
    b[0] = pos_cg[0] - k_x * pos_cr[0]
    b[1] = pos_cg[1] - k_y * pos_cr[1]
    # 计算转换坐标
    pos_g = [0.0,0.0]
    pos_g[0] = k_x * pos_r[0] + b[0]
    pos_g[1] = k_y * pos_r[1] + b[1]
    return pos_g

def start_detect_daemon(self):
        """启动守护进程"""
        from state_detect import state_detect_worker
        self.state_daemon = multiprocessing.Process(
            target=state_detect_worker,
            args=(self.task_queue, self.result_q),
            daemon=True
        )
        self.state_daemon.start()

def stop_detect_daemon(self):
    """停止守护进程"""
    if self.state_daemon and self.state_daemon.is_alive():
        self.task_queue.put((None, "STOP"))  # 发送终止信号
        self.state_daemon.join(timeout=5)

if __name__ == "__main__":
    # URTrackerConfig = Config()
    use_tobii = True
    use_robot = True
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

    
    # pipe_comm.start()

    ''' 初始化变量'''
    
    adaptive_param = {"distraction_freq": 1, "distraction_strength": 1, "param2": None, "Param3": None}  # 自适应游戏参数
    '''定义机器人相关参数'''
    # 注意需要与admittance 同步修改
    cursor_pos = [0.0, 0.0]
    target_pos = [0.0, 0.0]
    modetype = 1  # UR5控制参数的模式
    r_g = 4.0  # 游戏轨迹半径
    r_r = 0.1
    pos_cg = [0.0, 0.0]
    pos_cr = (0.625, -0.050)

    '''定义多线程数据传输队列'''
    Data_r2g_q = queue.Queue(maxsize=1)
    Data_g2r_q = queue.Queue(maxsize=1)
    command_q = queue.Queue(maxsize=1)
    
    Data_unity2py_q = queue.Queue(maxsize=1)
    Data_py2unity_q = queue.Queue(maxsize=1)
    read_target_q = queue.Queue(maxsize=1)
    send_cursor_q = queue.Queue(maxsize=1)
    # command_q = queue.Queue(maxsize=1)
    multiprocessing.set_start_method('spawn')
    # manager = multiprocessing.Manager()
    result_q = multiprocessing.Queue(maxsize=1)
    task_queue = multiprocessing.Queue(maxsize=1)
    state_daemon = None
    if not state_daemon or not state_daemon.is_alive():
        start_detect_daemon()
    # 定义数据 Dataframe
    data_columns = ['time', 'target_x', 'target_y', 'cursor_x', 'cursor_y', 'Hex_x', 'Hex_y', 'Hex_z','force_norm', 'linear_x', 'linear_y', 'linear_z',
                    'pose_x', 'pose_y', 'pose_z', 'pose_rx', 'pose_ry', 'pose_rz','Gaze_x','Gaze_y', 'stage_id', 'congnitive_flag', 'motor_flag', 'I_FB', 'I_ASST']

    df = pd.DataFrame(columns=data_columns)

    # 在游戏开始时启动模式识别守护进程
    # if not self.state_daemon or not self.state_daemon.is_alive():
    #     self.start_detect_daemon()

    '''初始化机器人控制器'''
    if use_robot:
        ur5_controller = Admittance()
        ur5_controller_t = threading.Thread(target=ur5_controller.run_test, args=(Data_g2r_q, Data_r2g_q, command_q),daemon=True)
        # 启动 UR5 控制线程
        ur5_controller_t.start()

    ''' 启动命名管道通信 '''
    unity2py_q = queue.Queue(maxsize=2)
    target_pipe = r'\\.\pipe\UnityToPythonPipe'  # 读取目标坐标
    cursor_pipe = r'\\.\pipe\PythonToUnityPipe'  # 发送光标坐标
    pipe_comm = PipeCommunication(target_pipe, cursor_pipe)

    threading.Thread(target=pipe_comm.read_target_coordinates, args=read_target_q, daemon=True).start()
    threading.Thread(target=pipe_comm.send_cursor_coordinates, args=send_cursor_q, daemon=True).start()


    while True:
        # 接收target数据from unity
        if not read_target_q.empty():
            target_pos = read_target_q.get()
        target_pos_r = pos_g2r(r_r, r_g, pos_cg, pos_cr, target_pos)
        # 发送target坐标到UR5 controller
        Data_g2r_q.put(target_pos_r)
        # 接收数据from UR5 controller
        if  not Data_r2g_q.empty():
            Data_r2g = Data_r2g_q.get()
        else:
            # 使用默认值或等待一段时间
            Data_r2g = {
                'time':time.time(), 'Hex_x': 0.0, 'Hex_y': 0.0, 'Hex_z': 0.0,
                'force_norm': 0.0, 'linear_x': 0.0, 'linear_y': 0.0, 'linear_z': 0.0,
                'pose_x': 0.725, 'pose_y': -0.050, 'pose_z': 0.0,
                'pose_rx': 0.0, 'pose_ry': 0.0, 'pose_rz': 0.0
            }
        pos_r = [Data_r2g['pose_x'], Data_r2g['pose_y']]
        cursor_pos = pos_r2g(r_r, r_g, pos_cg, pos_cr, pos_r)
        send_cursor_q.put(cursor_pos)
        if use_tobii:
            left_gaze = latest_gaze_data['left_gaze_point_on_display_area']
            right_gaze = latest_gaze_data['right_gaze_point_on_display_area']

            # 平均左右眼的注视点
            # TODO: 添加从unity json获取当前屏幕像素的长宽
            gaze_point_x = ((left_gaze[0] + right_gaze[0]) / 2-0.5) * 19.2 if left_gaze[0] > 0 and right_gaze[0] > 0 else None
            gaze_point_y = ((left_gaze[1] + right_gaze[1]) / 2-0.5) * 10.8 if left_gaze[1] > 0 and right_gaze[1] > 0 else None
            pos_gaze = [gaze_point_x, -gaze_point_y]

        '''记录所有数据'''
        df.loc[len(df.index)]=[time.time(), target_pos[0], target_pos[1], cursor_pos[0], cursor_pos[1],
                                        Data_r2g['Hex_x'], Data_r2g['Hex_y'], 
                                        Data_r2g['Hex_z'], Data_r2g['force_norm'], Data_r2g['linear_x'],
                                        Data_r2g['linear_y'], Data_r2g['linear_z'],
                                        Data_r2g['pose_x'],Data_r2g['pose_y'],Data_r2g['pose_z'],
                                        Data_r2g['pose_rx'],Data_r2g['pose_ry'],
                                        Data_r2g['pose_rz'], pos_gaze[0], pos_gaze[1], stage_id, congnitive_identify_flag, motor_identify_flag, I_FB, I_ASST]
                                        # Data_r2g['pose_rz'], pos_gaze[0], pos_gaze[1]]
                    
            


        



 
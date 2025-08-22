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
from bandpass_filter import robust_bandpass_filter
import datetime  # 添加这一行
from MCTSControlThread01 import mcts_thread_function

# 添加键盘监听相关导入
try:
    import msvcrt  # Windows系统
except ImportError:
    try:
        import tty, termios  # Unix/Linux系统
        import select
        import sys as sys_module
    except ImportError:
        pass

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

def calculate_deviations(data_queue, result_queue):
    """
    计算轨迹偏差的线程函数
    每0.5秒计算过去1秒窗口内的L1偏差
    """
    while True:
        try:
            # 从队列获取数据
            recent_data = None
            try:
                # 非阻塞获取最新的数据
                while True:
                    recent_data = data_queue.get_nowait()
            except queue.Empty:
                pass
            
            # 如果有数据则进行计算
            if recent_data and len(recent_data) >= 150:  # 确保有足够的数据点
                # 转换为numpy数组便于计算
                target_x = np.array([row["target_x"] for row in recent_data])
                target_y = np.array([row["target_y"] for row in recent_data])
                cursor_x = np.array([row["cursor_x"] for row in recent_data])
                cursor_y = np.array([row["cursor_y"] for row in recent_data])
                gaze_x = np.array([row["Gaze_x"] for row in recent_data])
                gaze_y = np.array([row["Gaze_y"] for row in recent_data])
                # filtered_cursor_x = robust_bandpass_filter(cursor_x, 0.1, 10.0, 50.0, method='mirror')
                # filtered_cursor_y = robust_bandpass_filter(cursor_y, 0.1, 10.0, 50.0, method='mirror')
                # filtered_gaze_x = robust_bandpass_filter(gaze_x, 0.1, 10.0, 50.0, method='mirror')
                # filtered_gaze_y = robust_bandpass_filter(gaze_y, 0.1, 10.0, 50.0, method='mirror')
                
                # 计算L1偏差
                # 机器人轨迹与目标轨迹的偏差
                # e_m = np.mean(np.abs(target_x - filtered_cursor_x) + np.abs(target_y - filtered_cursor_y))
                e_m = np.mean(np.sqrt((target_x - cursor_x)**2 + (target_y - cursor_y)**2))
                
                # 眼动轨迹与目标轨迹的偏差
                # e_c = np.mean(np.abs(target_x - filtered_gaze_x) + np.abs(target_y - filtered_gaze_y))
                e_c = np.mean(np.sqrt((target_x - gaze_x)**2 + (target_y - gaze_y)**2))
                
                # 将结果放入队列
                result_queue.put({ 
                    'e_m': e_m,
                    'e_c': e_c,
                    'timestamp': time.time()
                })
                
            # 每0.5秒计算一次
            time.sleep(0.2)
            
        except Exception as e:
            logging.error(f"计算偏差时出错: {e}")
            time.sleep(0.5)

def save_data_to_excel(df_window, filename=None):
    """保存数据到Excel文件"""
    if not df_window:
        logging.warning("数据窗口为空，无法保存数据")
        return
    
    # 如果没有提供文件名，则生成一个带时间戳的文件名
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_data_{timestamp}.xlsx"
    
    try:
        # 将数据转换为DataFrame
        df = pd.DataFrame(df_window)
        
        # 保存到Excel文件
        df.to_excel(filename, index=False)
        logging.info(f"数据已保存到 {filename}")
        return filename
    except Exception as e:
        logging.error(f"保存数据到Excel时出错: {e}")
        return None

def check_for_q_key():
    """检查是否按下了 'q' 键"""
    try:
        # Windows系统
        if os.name == 'nt':
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                return key == 'q'
        # Unix/Linux系统
        else:
            if select.select([sys_module.stdin], [], [], 0) == ([sys_module.stdin], [], []):
                key = sys_module.stdin.read(1).lower()
                return key == 'q'
    except:
        pass
    return False


def get_admittance_params_by_assistance_level(assistance_level):
    """
    根据辅助级别返回对应的导纳控制参数
    根据 cc_admittance_2022.py 中的模式定义:
    - high assistance (高辅助): mode1 - K=[400,400,400], D=[80,80,80], M=[40,40,40]
    - medium assistance (中等辅助): mode2 - K=[250,250,250], D=[80,80,80], M=[40,40,40]
    - low assistance (低辅助): mode3 - K=[5,5,5], D=[80,80,80], M=[1,1,1]
    """
    if assistance_level == 'high':
        return {'K': np.array([400, 400, 400]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    elif assistance_level == 'medium':
        return {'K': np.array([250, 250, 250]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    elif assistance_level == 'low':
        return {'K': np.array([5, 5, 5]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}
    else:
        # 默认返回中等辅助参数
        return {'K': np.array([250, 250, 250]), 'D': np.array([80, 80, 80]), 'M': np.array([40, 40, 40])}

def interpolate_params(start_params, end_params, progress):
    """
    在两个参数集之间进行插值
    progress: 0.0 到 1.0 之间的值，表示插值进度
    """
    interpolated = {}
    for key in ['K', 'D', 'M']:
        interpolated[key] = start_params[key] + progress * (end_params[key] - start_params[key])
    return interpolated


if __name__ == "__main__":
    setup_logger()
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
    # 创建队列用于识别线程间通信
    data_queue = queue.Queue(maxsize=1)  # 只保留最新的数据副本
    deviation_result_queue = queue.Queue(maxsize=1)
    mcts_in_q = queue.Queue(maxsize=1)
    mcts_out_q = queue.Queue(maxsize=1)

    # 识别参数
    tau_cog = 2.5
    tau_mot = 1.8
    # MCTS动作相关变量
    mcts_difficulty = 'medium'  # 默认难度
    mcts_feedback = 'medium'    # 默认反馈
    mcts_assistance = 'medium'  # 默认辅助

    # 渐变切换相关变量
    is_transitioning = False  # 是否正在进行参数渐变切换
    transition_start_time = None
    transition_duration = 1.0  # 渐变持续时间（秒）
    start_params = None  # 渐变开始时的参数
    target_params = None  # 渐变目标参数
    last_mcts_assistance = 'medium'
    
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

    # 启动偏差计算线程
    deviation_thread = threading.Thread(target=calculate_deviations, args=(data_queue, deviation_result_queue), daemon=True)
    deviation_thread.start()
    
    # # 启动状态识别子进程
    # task_q, result_q = multiprocessing.Queue(), multiprocessing.Queue()
    # detect_proc = multiprocessing.Process(target=state_detect_worker, args=(task_q, result_q), daemon=True)
    # detect_proc.start()

    # 启动MCTS线程
    mcts_thread = threading.Thread(target=mcts_thread_function, args=(mcts_in_q, mcts_out_q), daemon=True)
    mcts_thread.start()

    # 等待所有设备线程就绪
    time.sleep(2)
    

    # 添加循环频率控制
    target_frequency = 50  # 50Hz
    target_period = 1.0 / target_frequency  # 0.02秒
    last_loop_time = time.time()
    start_time = time.time()
    last_mcts_time = time.time()
    # 用于控制数据发送频率的计数器
    data_send_counter = 0
    mct_data_send_counter = 0

    # Unix/Linux系统需要设置终端为非阻塞模式
    if os.name != 'nt':
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            pass

    logging.info("程序已启动，按 'q' 键保存数据并退出")

    try:
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
            
            # 检查是否按下了 'q' 键
            if check_for_q_key():
                logging.info("检测到 'q' 键，准备保存数据并退出...")
                break
            
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
                
            except zmq.Again:
                logging.warning('未接收到target')
                time.sleep(0.001)  # 1ms
            except zmq.ZMQError as e:
                logging.error(f"接收Unity数据时ZMQ错误: {e}")
            except json.JSONDecodeError as e:
                logging.error(f"解析Unity数据JSON失败: {e}")
            except Exception as e:
                logging.error(f"接收Unity数据时未知错误: {e}")
            try:
                g2r_q.put_nowait(target_r)
            except:
                pass

            # 每10个循环发送一次数据副本给计算线程（约每0.2秒）
            data_send_counter += 1
            if data_send_counter >= 25:
                # 只发送最近50个点的数据
                if len(df_window) >= 150:
                    try:
                        # 发送最近50个点的数据，避免阻塞
                        data_queue.put_nowait(df_window[-150:])
                    except queue.Full:
                        pass  # 队列满时忽略
                data_send_counter = 0
            
            # 检查是否有新的偏差计算结果
            try:
                while not deviation_result_queue.empty():
                    result = deviation_result_queue.get_nowait()
                    e_m = result['e_m']
                    e_c = result['e_c']
                    logging.info(f"更新偏差: e_m={e_m:.4f}, e_c={e_c:.4f}")
                    if e_c < tau_cog and e_m < tau_mot:
                        state_flag = 0  #nAnM
                    elif e_c < tau_cog and e_m >= tau_mot:
                        state_flag = 1  #nAaM
                    elif e_c >= tau_cog and e_m < tau_mot:
                        state_flag = 2  #aAnM
                    else:
                        state_flag = 3  #aAaM
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"获取偏差结果时出错: {e}")

            # 每3秒发送一次状态数据副本给mcts线程
            mct_data_send_counter += 1
            if mct_data_send_counter >= 150:
                
                mcts_data = {
                    'state_flag':state_flag,
                    'e_c':e_c,
                    'e_m':e_m,
                    'timestamp':current_time
                }
                try:
                    mcts_in_q.put_nowait(mcts_data)
                    logging.info(f"向MCTS发送数据: state_flag={state_flag}, e_c={e_c:.4f}, e_m={e_m:.4f}")
                except queue.Full:
                    pass
                finally:
                    mct_data_send_counter = 0 
            if not mcts_out_q.empty():
                mcts_result = mcts_out_q.get()
                action = mcts_result['action']
                logging.info(f"接收到MCTS输出: 难度={action.difficulty}, 反馈={action.feedback}, 辅助={action.assistance}")
                # 这里可以根据MCTS的输出执行相应的操作
                # 更新MCTS动作状态
                mcts_difficulty = action.difficulty
                mcts_feedback = action.feedback
                mcts_assistance = action.assistance
                 # 检查辅助级别是否发生变化，如果变化则开始渐变切换
                if mcts_assistance != last_mcts_assistance and not is_transitioning:
                    # 获取新的导纳控制参数
                    target_params = get_admittance_params_by_assistance_level(mcts_assistance)
                    
                    # 开始渐变切换过程
                    is_transitioning = True
                    transition_start_time = current_time
                    start_params = get_admittance_params_by_assistance_level(last_mcts_assistance)
                    
                    logging.info(f"开始渐变切换辅助级别: {last_mcts_assistance} -> {mcts_assistance}")
                    last_mcts_assistance = mcts_assistance
            # 处理参数渐变切换
            if is_transitioning:
                elapsed_transition_time = current_time - transition_start_time
                progress = min(1.0, elapsed_transition_time / transition_duration)
                
                # 在当前参数和目标参数之间插值
                interpolated_params = interpolate_params(start_params, target_params, progress)
                
                # 发送插值后的参数到UR5控制器
                try:
                    cmd_q.put_nowait({
                        'type': 'update_params',
                        'K': interpolated_params['K'],
                        'D': interpolated_params['D'],
                        'M': interpolated_params['M']
                    })
                except queue.Full:
                    pass
                # 如果渐变完成，结束切换过程
                if progress >= 1.0:
                    is_transitioning = False
                    logging.info(f"辅助级别渐变切换完成: {last_mcts_assistance}")


            # 保存数据
            row = {"time":  time.time()-start_time, "target_x": target["x"], "target_y": target["y"],
                "cursor_x": cursor_g[0], "cursor_y": cursor_g[1],
                "Hex_x": state["Hex_x"], "Hex_y": state["Hex_y"],
                "force_norm": state["force_norm"], "linear_x": state["linear_x"],
                "linear_y": state["linear_y"], "linear_z": state["linear_z"],
                "pose_x": state["pose_x"], "pose_y": state["pose_y"],
                "pose_z": state["pose_z"], "pose_rx": state["pose_rx"],
                "pose_ry": state["pose_ry"], "pose_rz": state["pose_rz"],
                "Gaze_x": pos_gaze[0], "Gaze_y": pos_gaze[1],
                "stage_id":stage_id, "state_flag": state_flag,"e_c": e_c, "e_m":e_m,
                "I_FB": I_FB, "I_ASST": I_ASST,
                "mcts_difficulty": mcts_difficulty, "mcts_feedback": mcts_feedback, "mcts_assistance": mcts_assistance
            }
            df_window.append(row)



            ''' 输出数据 '''
            # 向Unity发送cursor数据
            cursor_data = {'x':cursor_g[0],'y':cursor_g[1]}
            cursor_pub_socket.send_multipart([b"cursor", json.dumps(cursor_data).encode('utf-8')])
            logging.debug('发送光标：%f, %f',cursor_g[0], cursor_g[1])
            # 每10个循环发送一次指标数据到Unity（约每0.2秒）
            # metrics_send_counter += 1
            # if metrics_send_counter >= 10:
                # 发布e_c和e_m数据，使用"metrics" topic
            metrics_data = {
                'e_c': float(e_c),
                'e_m': float(e_m),
                'timestamp': current_time
            }
            try:
                cursor_pub_socket.send_multipart([b"metrics", json.dumps(metrics_data).encode('utf-8')])
            except Exception as e:
                logging.error(f"发布指标数据时出错: {e}")
            
            # 发布MCTS动作数据，使用"mcts_action" topic
            if 'mcts_difficulty' in locals() and 'mcts_feedback' in locals() and 'mcts_assistance' in locals():
                mcts_action_data = {
                    'difficulty': mcts_difficulty,
                    'feedback': mcts_feedback,
                    'assistance': mcts_assistance,
                    'timestamp': current_time
                }
                try:
                    cursor_pub_socket.send_multipart([b"mcts_action", json.dumps(mcts_action_data).encode('utf-8')])
                except Exception as e:
                    logging.error(f"发布MCTS动作数据时出错: {e}")
                
                # metrics_send_counter = 0
            



            

    except Exception as e:
        logging.error(f"程序运行时发生异常: {e}")
        traceback.print_exc()
    finally:
        logging.info("正在保存实验数据...")
        filename = save_data_to_excel(df_window)
        if filename:
            logging.info(f"数据已保存到: {filename}")
        else:
            logging.error("数据保存失败")
        logging.info("程序已退出")
        
        # 恢复Unix/Linux系统的终端设置
        if os.name != 'nt':
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
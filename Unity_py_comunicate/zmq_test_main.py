#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' 测试与ZmqManager.cs的ZMQ通信 '
__author__ = 'Test'
__date__ = '2024/01/01'
__version__ = '1.0'

import sys
import os
import time
import numpy as np
import zmq
import json
import threading
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

class ZMQCommunicationTest:
    def __init__(self):
        # 初始化ZMQ上下文
        self.context = zmq.Context()
        
        # 创建发布套接字（对应ZmqManager.cs的订阅者）
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:6001")  # 与ZmqManager.cs的接收端口一致
        self.pub_socket.setsockopt(zmq.SNDHWM, 10)
        
        logging.info("已绑定发布套接字到 tcp://*:6001")
        
        # 创建订阅套接字（对应ZmqManager.cs的发布者）
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.bind("tcp://localhost:6005")  # 与ZmqManager.cs的发布端口一致
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息
        self.sub_socket.setsockopt(zmq.RCVHWM, 10)
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        logging.info("已连接订阅套接字到 tcp://localhost:6005")
        
        # 控制标志
        self.running = True
        
        # 接收线程
        self.receive_thread = threading.Thread(target=self.receive_messages)
        self.receive_thread.daemon = True
        
        # 测试数据
        self.cursor_position = {"x": 0.0, "y": 0.0}
        self.metrics_data = {"e_c": 0.0, "e_m": 0.0}
        self.mcts_action = {"difficulty": "medium", "feedback": "medium", "assistance": "medium"}
    
    def start(self):
        """启动测试"""
        logging.info("ZMQ通信测试开始")
        self.receive_thread.start()
        self.main_loop()
    
    def stop(self):
        """停止测试"""
        self.running = False
        time.sleep(0.1)
        self.context.term()
        logging.info("ZMQ通信测试结束")
    
    def receive_messages(self):
        """接收来自ZmqManager.cs的消息"""
        while self.running:
            try:
                # 非阻塞接收消息
                message = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
                if message:
                    try:
                        data = json.loads(message)
                        if "topic" in data:
                            topic = data["topic"]
                            # logging.info(f"收到Unity消息 - Topic: {topic}, 内容: {data}")
                            
                            if topic == "TargetPosition":
                                # 处理目标位置消息
                                if "x" in data and "y" in data:
                                    #logging.info(f"目标位置: x={data['x']:.4f}, y={data['y']:.4f}")
                                    pass
                            elif topic == "TaskStage":
                                # 处理任务阶段消息
                                if "stage" in data:
                                    logging.info(f"任务阶段变更为: {data['stage']}")
                        else:
                            logging.warning(f"收到无效消息格式: {message}")
                    except json.JSONDecodeError as e:
                        logging.error(f"解析JSON消息失败: {e}")
            except zmq.Again:
                # 没有消息可接收，正常情况
                pass
            except zmq.ZMQError as e:
                logging.error(f"ZMQ接收错误: {e}")
            except Exception as e:
                logging.error(f"接收消息时发生未知错误: {e}")
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)
    
    def send_cursor_position(self):
        """发送光标位置消息"""
        try:
            data = {
                "x": self.cursor_position["x"],
                "y": self.cursor_position["y"]
            }
            self.pub_socket.send_multipart([b"cursor", json.dumps(data).encode()])
            logging.info(f"发送光标位置: x={data['x']:.4f}, y={data['y']:.4f}")
        except Exception as e:
            logging.error(f"发送光标位置失败: {e}")
    
    def send_metrics(self):
        """发送指标数据消息"""
        try:
            data = {
                "e_c": self.metrics_data["e_c"],
                "e_m": self.metrics_data["e_m"],
                "timestamp": time.time()
            }
            self.pub_socket.send_multipart([b"metrics", json.dumps(data).encode()])
            logging.info(f"发送指标数据: e_c={data['e_c']:.4f}, e_m={data['e_m']:.4f}")
        except Exception as e:
            logging.error(f"发送指标数据失败: {e}")
    
    def send_mcts_action(self):
        """发送MCTS动作消息"""
        try:
            data = {
                "difficulty": self.mcts_action["difficulty"],
                "feedback": self.mcts_action["feedback"],
                "assistance": self.mcts_action["assistance"],
                "timestamp": time.time()
            }
            self.pub_socket.send_multipart([b"mcts_action", json.dumps(data).encode()])
            logging.info(f"发送MCTS动作: 难度={data['difficulty']}, 反馈={data['feedback']}, 辅助={data['assistance']}")
        except Exception as e:
            logging.error(f"发送MCTS动作失败: {e}")
    
    def main_loop(self):
        """主交互循环"""
        logging.info("\nZMQ通信测试控制台")
        logging.info("====================")
        logging.info("命令列表:")
        logging.info("1 - 发送光标位置")
        logging.info("2 - 发送指标数据")
        logging.info("3 - 发送MCTS动作")
        logging.info("4 - 更新光标位置")
        logging.info("5 - 更新指标数据")
        logging.info("6 - 更新MCTS动作")
        logging.info("q - 退出测试")
        logging.info("====================\n")
        
        while self.running:
            command = input("请输入命令: ").strip().lower()
            
            if command == 'q':
                self.running = False
            elif command == '1':
                self.send_cursor_position()
            elif command == '2':
                self.send_metrics()
            elif command == '3':
                self.send_mcts_action()
            elif command == '4':
                try:
                    x = float(input("请输入光标X坐标: "))
                    y = float(input("请输入光标Y坐标: "))
                    self.cursor_position = {"x": x, "y": y}
                    logging.info(f"光标位置已更新为: x={x:.4f}, y={y:.4f}")
                except ValueError:
                    logging.error("无效的坐标值，请输入数字")
            elif command == '5':
                try:
                    e_c = float(input("请输入e_c值: "))
                    e_m = float(input("请输入e_m值: "))
                    self.metrics_data = {"e_c": e_c, "e_m": e_m}
                    logging.info(f"指标数据已更新为: e_c={e_c:.4f}, e_m={e_m:.4f}")
                except ValueError:
                    logging.error("无效的指标值，请输入数字")
            elif command == '6':
                difficulty = input("请输入难度级别 (easy/medium/hard): ").strip().lower()
                feedback = input("请输入反馈级别 (low/medium/high): ").strip().lower()
                assistance = input("请输入辅助级别 (low/medium/high): ").strip().lower()
                
                if difficulty in ['easy', 'medium', 'hard'] and \
                   feedback in ['low', 'medium', 'high'] and \
                   assistance in ['low', 'medium', 'high']:
                    self.mcts_action = {
                        "difficulty": difficulty,
                        "feedback": feedback,
                        "assistance": assistance
                    }
                    logging.info(f"MCTS动作已更新: 难度={difficulty}, 反馈={feedback}, 辅助={assistance}")
                else:
                    logging.error("无效的参数值")
            else:
                logging.error("无效的命令，请重新输入")

if __name__ == "__main__":
    test = ZMQCommunicationTest()
    try:
        test.start()
    except KeyboardInterrupt:
        logging.info("接收到中断信号")
    finally:
        test.stop()
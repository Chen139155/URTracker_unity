#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import json

def main():
    # 创建ZeroMQ上下文
    context = zmq.Context()
    
    # 创建Subscriber套接字
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    # 连接并订阅
    subscriber.bind("tcp://localhost:6005")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("开始接收Unity副目标位置数据，按 Ctrl+C 停止...")
    
    try:
        while True:
            try:
                # 接收并解析数据
                message = subscriber.recv_string()
                data = json.loads(message)
                print(f"副目标位置: x={data['x']:.4f}, y={data['y']:.4f}")
            except Exception as e:
                print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        subscriber.close()
        context.term()

if __name__ == "__main__":
    main()
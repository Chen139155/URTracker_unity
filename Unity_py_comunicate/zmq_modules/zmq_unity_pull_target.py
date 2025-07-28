# zmq_modules/zmq_unity_pull_target.py
import zmq
import json

def get_target_generator():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)  # 设置linger为0确保立即释放
    socket.bind("tcp://*:6005")  # Unity连接这个端口发送目标
    
    try:
        while True:
            msg = socket.recv_json()
            yield msg  # {"x": float, "y": float}
    except Exception as e:
        print(f"接收目标数据错误: {e}")
    finally:
        socket.close()
        context.term()  # 确保关闭context
# context = zmq.Context()
# socket = context.socket(zmq.PULL)
# socket.bind("tcp://*:6005")  # Unity连接这个端口发送目标

# def get_target_generator():
#     while True:
#         msg = socket.recv_json()
#         yield msg  # {"x": float, "y": float}

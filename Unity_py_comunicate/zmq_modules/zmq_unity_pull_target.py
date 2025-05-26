# zmq_modules/zmq_unity_pull_target.py
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:6000")  # Unity连接这个端口发送目标

def get_target_generator():
    while True:
        msg = socket.recv_json()
        yield msg  # {"x": float, "y": float}

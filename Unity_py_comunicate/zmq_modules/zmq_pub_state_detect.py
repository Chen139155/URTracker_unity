# zmq_modules/zmq_pub_state_detect.py
import zmq

# 不再自动创建上下文
context = None
socket = None

def init_publisher(external_context=None):
    global context, socket
    context = external_context or zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:6002")

def broadcast_state_result(data):
    if socket:
        socket.send_json(data)  # e.g. {'state_flag': 1, 'e_cog_norm': 0.3, 'e_mot_norm': 0.1}

# zmq_modules/zmq_pub_state_detect.py
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:6002")

def broadcast_state_result(data):
    socket.send_json(data)  # e.g. {'state_flag': 1, 'e_cog_norm': 0.3, 'e_mot_norm': 0.1}

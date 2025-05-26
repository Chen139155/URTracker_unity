# zmq_modules/zmq_unity_push_cursor.py
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://localhost:6001")  # Unity监听此端口

def send_cursor_position(x, y):
    socket.send_json({"x": x, "y": y})

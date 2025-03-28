import URBasic.robotModel
import URBasic.urScriptExt
# import test_main as Ur
import hex              # 与HEX通信
import numpy as np
# import threading
# import socket
import time
import URBasic

if __name__ == '__main__':
    # initialise robot with URBasic
    print("initialising robot")
    host = '10.168.2.209'
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=host,robotModel=robotModel)

    robot.reset_error()
    print("robot initialised")
    time.sleep(1)

    # Move Robot to the midpoint of the lookplane
    # robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY )
    while(True):
        pose = robot.get_actual_tcp_pose()
        print('tcp pose:',pose)
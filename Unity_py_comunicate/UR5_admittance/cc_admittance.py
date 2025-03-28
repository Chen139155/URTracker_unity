import URBasic.robotModel
import URBasic.urScriptExt
# import test_main as Ur
import hex              # 与HEX通信
import numpy as np
import threading
import socket
import time
import URBasic
import pandas as pd
import sys
# import PyLocationServer as PLS
# import struct
# import util
# import rtde

class Admittance:
    
    def __init__(self,desired_pose=np.array([0.400,-0.110,0.528,0,3.141,0]),
                 arm_max_vel=0.5,
                 arm_max_acc=0.5,
                 M=np.array([0.2,0.2,0.2]),
                 D=np.array([8,8,8]),
                 K=np.array([0.2,0.2,0.2]),
                 frequency=20):
        self.M_=M
        self.D_=D
        self.K_=K
        self.desired_pose_=desired_pose
        self.arm_max_vel_=arm_max_vel
        self.arm_max_acc_=arm_max_acc
        self.fixed_time_=1/frequency
        self.duration_=self.fixed_time_
        #初始化机器人 client
        self.URHOST_='10.168.2.209'
        self.URPORT_=30003
        print("initialising robot")
        robotModel = URBasic.robotModel.RobotModel()
        self.UR_ = URBasic.urScriptExt.UrScriptExt(host=self.URHOST_,robotModel=robotModel)
        self.UR_.reset_error()
        print("robot initialised")  
        time.sleep(1)
        # self.UR_ = Ur.UrRobot(self.URHOST_,self.URPORT_) # 与UR5通信
        #server
        # 用于与unity通信的server
        # self.pyserver = PLS.PyServer()
        # 初始化类变量
        self.arm_pose_ = np.zeros(6) #当前位姿，6维浮点数
        self.arm_position_ = self.arm_pose_[0:3]
        self.arm_tcp_speed_ = np.zeros(6)
        self.speed_norm_ = 0.0
        self.force_external_ = np.zeros(3) 
        self.force_norm_ = 0.0
        # 初始化累积量
        self.arm_desired_twist_adm_ = np.zeros(3)
        self.admittance_desired_pose_ = self.desired_pose_

        # 初始化标志位
        self.event_flag_ = 'start'

    def run(self):

        self.UR_.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
        time.sleep(1) # just a short wait to make sure everything is initialised
        print('Runing the admittance control loop ...........')
        # 记录数据
        data_columns = ['time', 'Hex_x', 'Hex_y', 'Hex_z', 'linear_x', 'linear_y', 'linear_z',
                         'pose_x', 'pose_y', 'pose_z', 'pose_rx', 'pose_ry', 'pose_rz']
        df = pd.DataFrame(columns=data_columns)

        t_flag=threading.Thread(target=self.event_flag_get)
        t_flag.start()
        # 开启从向unity发送消息的client线程
        t_client = threading.Thread(target=self.client_to_unity_thread)
        t_client.start()

        print("threads on")
        start_time = time.time()
        while(True):   # 记得改成通信成功
            
            if self.event_flag_ == 'quit':
                df.to_excel('./output/'+ str(time.time())+'.xlsx')
                
                sys.exit() #直接退出可能出问题，记得修改

            self.get_state()  #获取力与位置信息
            df.loc[len(df.index)]=[time.time(), self.force_external_[0], self.force_external_[1], self.force_external_[2],
                                   self.arm_tcp_speed_[0],self.arm_tcp_speed_[1],self.arm_tcp_speed_[2],
                                   self.arm_pose_[0],self.arm_pose_[1],self.arm_pose_[2],self.arm_pose_[3],self.arm_pose_[4],
                                   self.arm_pose_[5]]
            self.compute_admittance()
            self.send_commands_to_robot()
            print(time.time(),'\n'+ 'pose: ',self.arm_pose_,'\nforce: ',self.force_external_,'\ncontrl pose:',self.admittance_desired_pose_,'\nv: ',self.arm_tcp_speed_)
            
            
            # 计算时间差并补偿
            # start_time = time.time()
            elapsed_time = time.time()-start_time
            if elapsed_time < self.fixed_time_:
                time.sleep(self.fixed_time_-elapsed_time)
                self.duration_ = self.fixed_time_

            else:
                # print('continue')
                self.duration_ = elapsed_time
                # continue
            # self.duration_ = elapsed_time
            print('duration',self.duration_)
            start_time = time.time()

    def compute_admittance(self):
        s_time = time.time()
        error = self.arm_position_ - self.desired_pose_[0:3] #arm_position用并行程序获取？还是直接在run循环里面获取？
        coupling_wrench_arm = np.zeros(3)
        arm_desired_vel = self.arm_desired_twist_adm_
        coupling_wrench_arm = self.D_ * arm_desired_vel + self.K_ * error
        arm_desired_acc = (-coupling_wrench_arm + self.force_external_)/self.M_
        desired_acc_norm = np.linalg.norm(arm_desired_acc)
        if desired_acc_norm > self.arm_max_acc_:
            print(time.time(),': Admittance generates high arm accelration! acc norm:',desired_acc_norm)
            arm_desired_acc *= self.arm_max_acc_ / desired_acc_norm
        
        duration = self.duration_
        arm_desired_vel += arm_desired_acc * duration*0.3

        desired_vel_norm = np.linalg.norm(arm_desired_vel)
        if desired_vel_norm > self.arm_max_vel_:
            print(time.time(),': Admittance generates high arm vel! vel norm:', desired_vel_norm)
            arm_desired_vel *= self.arm_max_vel_ / desired_vel_norm
        self.arm_desired_twist_adm_ = arm_desired_vel 
        admittance_desired_position = self.admittance_desired_pose_[0:3]
        admittance_desired_position += arm_desired_vel*duration
        self.admittance_desired_pose_[0:3] = admittance_desired_position
        
        e_time = time.time()
        # print('compute_admittance 执行时间:',e_time-s_time)
        print('a=',arm_desired_acc,'v=',arm_desired_vel,'p=',admittance_desired_position)
        

    # 获取机械臂末端位姿与力传感器数据
    def get_state(self):
        s_time = time.time()
        # 末端位姿
        self.arm_pose_ = self.UR_.get_actual_tcp_pose() #当前位姿，x,y,z,rx,ry,rz 6维浮点数
        self.arm_position_ = self.arm_pose_[0:3]
        # self.pyserver.get_syn_position(self.arm_pose_[0:2])
        e_time_1 = time.time()
        # print('get_state position 执行时间:',e_time_1-s_time)
        # 末端速度
        self.arm_tcp_speed_ = self.UR_.get_actual_tcp_speed() # 当前速度x,y,z,rx,ry,rz
        self.speed_norm_ = np.linalg.norm(self.arm_tcp_speed_[0:2])
        # 力传感器数据
        external_force_data = np.asarray(hex.udp_get()) # fx,fy,fz,tx,ty,tz
        # self.force_external_ = external_force_data[0:2] # x0.3 y0.65 漂移补偿+死区
        f_x=external_force_data[0]
        f_y=external_force_data[1]-0.65
        if abs(f_x) < 1:
            f_x = 0
        if abs(f_y) < 1:
            f_y = 0
        self.force_external_[0:2] = np.array([-f_x,f_y])# 力传感器x轴与ur5 x轴反向
        self.force_norm_ = np.linalg.norm([self.force_external_[0:2]])
        e_time = time.time()
        # print('get_state hex 执行时间:',e_time-e_time_1)
     

    def send_commands_to_robot(self):
        s_time = time.time()
        self.UR_.set_realtime_pose(self.admittance_desired_pose_)
        time.sleep(0.001)
        e_time = time.time()
        # print('send_commands_to_robot执行时间:',e_time-s_time)

    def update_control_param(self,K,D,M):
        self.K_=K
        self.D_=D
        self.M_=M

    # 仿真
    def sim_run(self):
        print('Runing the admittance control loop ...........')

        while(True):   # 记得改成通信成功
            self.sim_get_state()  #获取力与位置信息
            self.compute_admittance()
            # self.send_commands_to_robot()
            print(time.time(),'\n'+ 'pose: ',self.arm_pose_,'\nforce: ',self.force_external_,'\ncontrl pose:',self.admittance_desired_pose_)
            start_time = time.time()
            
            # 计算时间差并补偿
            elapsed_time = time.time()-start_time
            if elapsed_time < self.fixed_time_:
                time.sleep(self.fixed_time_-elapsed_time)
            else:
                break
            start_time = time.time()
    
    def sim_get_state(self):
        # 末端位姿
        time.sleep(0.01)
        self.arm_pose_ = self.admittance_desired_pose_ #当前位姿，6维浮点数
        self.arm_position_ = self.arm_pose_[0:3]
        # 力传感器数据
        external_force_data = np.asarray(hex.udp_get()) # fx,fy,fz,tx,ty,tz
        # self.force_external_ = external_force_data[0:2] # x0.3 y0.65 漂移补偿+死区
        f_x=external_force_data[0]
        f_y=external_force_data[1]-0.3
        if abs(f_x) < 0.5:
            f_x = 0
        if abs(f_y) < 0.5:
            f_y = 0
        self.force_external_[0:2] = np.array([f_x,f_y]) 

    def event_flag_get(self):
        while True:
            temp_str = input()
            if temp_str == '1':
                self.event_flag_='quit'

    def client_to_unity_thread(self):
        while True:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建Socket的客户端
                client.connect(('127.0.0.1', 50088))  # 设置相对应的ip地址和端口号 我的IP:192.168.43.2 端口号：1111
            except:
                continue
            else:
                x = self.arm_position_[0]
                y = self.arm_position_[1]
                str_content = 'x=' + str(x) + ',y=' + str(y) +',f='+str(self.force_norm_)+',v='+str(self.speed_norm_)
                # message = str(time.time())+str(sync_robot_loc)  # 输入要发送的内容
                time.sleep(0.01)
                # print(str(time.time())+str(sync_robot_loc))
                client.send(str_content.encode('utf-8'))  # 发送
                client.close() #结束后关闭


if __name__ == '__main__':
    UR_control = Admittance()
    UR_control.run()




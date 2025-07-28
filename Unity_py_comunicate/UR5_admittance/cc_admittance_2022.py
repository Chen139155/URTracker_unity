import sys
import os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加完整的 URBasic 模块路径
sys.path.append(current_dir)
import URBasic
import URBasic.robotModel
import URBasic.urScriptExt
# import test_main as Ur
import hex              # 与HEX通信
import math
import numpy as np
import threading
import socket
import time
import pandas as pd
import queue
# import PyLocationServer as PLS
# import struct
# import util
# import rtde

class Admittance:
    
    def __init__(self,desired_pose=np.array([0.625,-0.050,0.400,0,3.141,0]),
                 arm_max_vel=0.5,
                 arm_max_acc=0.5,
                 M=np.array([40,40,40]),
                 D=np.array([80,80,80]),
                 K=np.array([250,250,250]),
                 frequency=20):
        self.M_=M
        self.D_=D
        self.K_=K
        self.w_r = 1 # 预设轨迹角速度
        self.R_c = 0.1 # 预设轨迹半径
        self.theta = 0.0 # 角度
        self.pos_cr = (0.400,-0.110)
        self.desired_pose_=desired_pose
        self.desired_pose_[0]=desired_pose[0] - self.R_c
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
        time.sleep(0.5)
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
        self.x_e = np.zeros(3)

        # 初始化标志位
        self.event_flag_ = 'mode1'
        self.runing = True

        # 数据输出队列
        self.data_output_queue_ = queue.Queue()

    def run(self):

        self.UR_.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
        time.sleep(1) # just a short wait to make sure everything is initialised
        print('Runing the admittance control loop ...........')
        # 记录数据
        data_columns = ['time', 'Hex_x', 'Hex_y', 'Hex_z', 'linear_x', 'linear_y', 'linear_z',
                         'pose_x', 'pose_y', 'pose_z', 'pose_rx', 'pose_ry', 'pose_rz']
        df = pd.DataFrame(columns=data_columns)

        # t_flag=threading.Thread(target=self.event_flag_get) # 游戏中禁用按键识别线程
        # t_flag.start()
        # 开启从向unity发送消息的client线程
        t_client = threading.Thread(target=self.client_to_unity_thread)
        # t_client.start()

        print("threads on")
        start_time = time.time()
        while(True):   # 记得改成通信成功
            
            if self.event_flag_ == 'quit':
                # df.to_excel('./output/'+ str(time.time())+'.xlsx')
                
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


    def run_test(self, data_g2r_q, data_r2g_q, command_q):
        '''力辅助模式测试'''
        self.UR_.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
        self.runing = True
        time.sleep(1) # just a short wait to make sure everything is initialised
        print('Runing the admittance control loop ...........')
        

        # t_flag=threading.Thread(target=self.event_flag_get, daemon=True)
        # t_flag.start()
        
        # print("threads on")
        start_time = time.time()
        
        while(self.runing):   # 记得改成通信成功
            
            # 处理命令队列
            if not command_q.empty():
                self.event_flag_ = command_q.get()
                print('command_q: ',self.event_flag_)
                if self.event_flag_ == 'quit':
                    self.runing = False
                    sys.exit() #直接退出可能出问题，记得修改
                elif self.event_flag_ == 'mode1':
                    self.update_mode(1)
                elif self.event_flag_ == 'mode2':
                    self.update_mode(2)
                elif self.event_flag_ == 'mode3':
                    self.update_mode(3)
                elif self.event_flag_ == 'stop':
                    continue

            if not data_g2r_q.empty():
                data_g2r = data_g2r_q.get()
                self.desired_pose_[0] = data_g2r[0]
                self.desired_pose_[1] = data_g2r[1]

            
            self.get_state()  #获取力与位置信息
            
            self.compute_admittance_test()
            self.send_commands_to_robot()
            # print(time.time(),'\n'+ 'pose: ',self.arm_pose_,'\nforce: ',self.force_external_,'\ncontrl pose:',self.admittance_desired_pose_,'\nv: ',self.arm_tcp_speed_)
            
            # 将数据放入数据队列
            data = {
                'time': time.time(),
                'Hex_x': self.force_external_[0],
                'Hex_y': self.force_external_[1],
                'Hex_z': self.force_external_[2],
                'force_norm': self.force_norm_,
                'linear_x': self.arm_tcp_speed_[0],
                'linear_y': self.arm_tcp_speed_[1],
                'linear_z': self.arm_tcp_speed_[2],
                'pose_x': self.arm_pose_[0],
                'pose_y': self.arm_pose_[1],
                'pose_z': self.arm_pose_[2],
                'pose_rx': self.arm_pose_[3],
                'pose_ry': self.arm_pose_[4],
                'pose_rz': self.arm_pose_[5],
                # 'Gaze_x': latest_gaze_data["left_gaze_point_on_display_area"][0] if latest_gaze_data else 0,
                # 'Gaze_y': latest_gaze_data["left_gaze_point_on_display_area"][1] if latest_gaze_data else 0,
                # 'stage_id': self.stage_id,
                # 'congnitive_flag': self.congnitive_flag,
                # 'motor_flag': self.motor_flag
            }
            data_r2g_q.put(data)
            
            # 计算时间差并补偿
            
            elapsed_time = time.time()-start_time
            if elapsed_time < self.fixed_time_:
                time.sleep(self.fixed_time_-elapsed_time)
                self.duration_ = self.fixed_time_

            else:
                # print('continue')
                self.duration_ = elapsed_time
                # continue
            # self.duration_ = elapsed_time
            # print('duration',self.duration_)
            self.theta += self.w_r*self.duration_
            if self.theta > math.pi*2:
                self.theta -= math.pi*2
            start_time = time.time()
        self.get_state()  #获取力与位置信息
        self.UR_.set_realtime_pose(self.arm_pose_)
        self.send_commands_to_robot()

    def compute_admittance_test(self):
        '''力辅助模式测试'''
        s_time = time.time()
        
        x = self.arm_position_
        x_d = self.desired_pose_[0:3]
        # x_e = x - x_d 
        x_e = self.x_e
        F_s = np.zeros(3)
        dx_e = self.arm_desired_twist_adm_
        F_s = self.D_ * dx_e + self.K_ * x_e
        F_ext = self.force_external_
        ddx_e = (-F_s + F_ext)/self.M_
        ddx_e_norm = np.linalg.norm(ddx_e)
        if ddx_e_norm > self.arm_max_acc_:
            # print(time.time(),': Admittance generates high arm accelration! acc norm:',ddx_e_norm)
            ddx_e *= self.arm_max_acc_ / ddx_e_norm
        
        delta_t = self.duration_
        dx_e += ddx_e * delta_t*0.3

        dx_e_norm = np.linalg.norm(dx_e)
        if dx_e_norm > self.arm_max_vel_:
            # print(time.time(),': Admittance generates high arm vel! vel norm:', dx_e_norm)
            dx_e *= self.arm_max_vel_ / dx_e_norm
        self.arm_desired_twist_adm_ = dx_e 
        self.x_e += dx_e*delta_t
        x_r = x_d+ self.x_e
        
        # admittance_desired_position = self.admittance_desired_pose_[0:3]
        # admittance_desired_position += dx_e*delta_t
        self.admittance_desired_pose_[0:3] = x_r
        
        e_time = time.time()
        # print('compute_admittance 执行时间:',e_time-s_time)
        # print('a=',ddx_e,'v=',dx_e,'p=',x_r)

    def compute_admittance(self):
        s_time = time.time()
        error = self.arm_position_ - self.desired_pose_[0:3] #arm_position用并行程序获取？还是直接在run循环里面获取？
        coupling_wrench_arm = np.zeros(3)
        arm_desired_vel = self.arm_desired_twist_adm_
        coupling_wrench_arm = self.D_ * arm_desired_vel + self.K_ * error
        arm_desired_acc = (-coupling_wrench_arm + self.force_external_)/self.M_
        desired_acc_norm = np.linalg.norm(arm_desired_acc)
        if desired_acc_norm > self.arm_max_acc_:
            # print(time.time(),': Admittance generates high arm accelration! acc norm:',desired_acc_norm)
            arm_desired_acc *= self.arm_max_acc_ / desired_acc_norm
        
        duration = self.duration_
        arm_desired_vel += arm_desired_acc * duration*0.3

        desired_vel_norm = np.linalg.norm(arm_desired_vel)
        if desired_vel_norm > self.arm_max_vel_:
            # print(time.time(),': Admittance generates high arm vel! vel norm:', desired_vel_norm)
            arm_desired_vel *= self.arm_max_vel_ / desired_vel_norm
        self.arm_desired_twist_adm_ = arm_desired_vel 
        admittance_desired_position = self.admittance_desired_pose_[0:3]
        admittance_desired_position += arm_desired_vel*duration
        self.admittance_desired_pose_[0:3] = admittance_desired_position
        
        e_time = time.time()
        # print('compute_admittance 执行时间:',e_time-s_time)
        # print('a=',arm_desired_acc,'v=',arm_desired_vel,'p=',admittance_desired_position)
        
        

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
        f_x=external_force_data[0]+0.1
        f_y=external_force_data[1]+0.3
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
        '''更新导纳控制参数 np.array([x,x,x])'''
        self.K_=K
        self.D_=D
        self.M_=M
    
    def update_mode(self, modetype:int):
        '''直接控制切换模式'''
        if modetype == 1:
            self.K_=np.array([5,5,5])
            self.D_=np.array([80,80,80])
            self.M_=np.array([1,1,1])
        elif modetype == 2:
            self.K_=np.array([250,250,250])
            self.D_=np.array([80,80,80])
            self.M_=np.array([40,40,40])
        elif modetype == 3:
            self.K_=np.array([400,400,400])
            self.D_=np.array([80,80,80])
            self.M_=np.array([40,40,40])
            

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
        '''通过键盘获取event_flag'''
        while True:
            temp_str = input()
            if temp_str == 'q':
                self.event_flag_='quit'
            elif temp_str == '1':
                self.event_flag_='mode1'
            elif temp_str == '2':
                self.event_flag_='mode2'
            elif temp_str == '3':
                self.event_flag_='mode3'

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

    def get_posxy(self):
        '''返回末端的xy轴坐标'''
        posxy = (self.arm_pose_[0],self.arm_pose_[1])
        return posxy
    
    def update_desired_pos(self, pos):
        '''从外部获取目标位置坐标'''
        self.desired_pose_[0] = pos[0]
        self.desired_pose_[1] = pos[1]

    def stop_runing(self):
        '''停止控制'''
        '''停止函数有问题，谨慎使用'''
        self.runing = False
        self.get_state()  #获取力与位置信息
        self.UR_.set_realtime_pose(self.arm_pose_)
        self.send_commands_to_robot()

if __name__ == '__main__':
    UR_control = Admittance()
    UR_control.run_test()




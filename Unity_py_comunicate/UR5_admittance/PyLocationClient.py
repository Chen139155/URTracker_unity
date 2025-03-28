import socket
import time


sync_robot_loc=[0.400,-0.110]

while True:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建Socket的客户端
            client.connect(('127.0.0.1', 50088))  # 设置相对应的ip地址和端口号 我的IP:192.168.43.2 端口号：1111
        except:
            continue
        else:
            x = sync_robot_loc[0]
            y = sync_robot_loc[1]
            str_content = 'x=' + str(x) + ',y=' + str(y)
            # message = str(time.time())+str(sync_robot_loc)  # 输入要发送的内容
            time.sleep(0.02)
            print(str(time.time())+str(sync_robot_loc))
            client.send(str_content.encode('utf-8'))  # 发送
            client.close() #结束后关闭
 
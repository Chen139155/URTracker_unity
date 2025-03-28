import socket
 
class PyServer:
    def __init__(self):
        # 创建socket对象并绑定IP端口
        self.s=socket.socket()
        self.s.bind(('127.0.0.1',50088))
        self.s.listen()
        self.sync_robot_loc=[0.400,-0.110]
    # sync_robot_loc=[12.3,13.6]

    def def_socket_thread(self):
        # global loop
        # loop = asyncio.get_event_loop()
        try:
            while True:
                # 接受一个客户端的连接请求
                c, addr = self.s.accept()
                content = self.read_from_client(c)
                if content.find('location') > -1:
                    # global sync_robot_loc
                    print('receive request')
                    print('sycn position=', self.sync_robot_loc)
                    x = self.sync_robot_loc[0]
                    y = self.sync_robot_loc[1]
                    str_content = 'x=' + str(x) + ',y=' + str(y)
                    c.send(str_content.encode('ascii'))
                    print('finish location send')
                else:
                    print('no request')
        except IOError as e:
            print(e.strerror)
        print('start socket thread!!!')
    
    
    def read_from_client(self,c):
        try:
            return c.recv(1024).decode('ascii')
        except IOError as e:
            # 如果异常的话可能就是会话中断 那么直接删除
            print(e.strerror)

    # 获取要发送的坐标信息（从外部输入）
    def get_syn_position(self,posxy):
        self.sync_robot_loc = posxy

 
if __name__=='__main__':
    location_server = PyServer()
    location_server.def_socket_thread()
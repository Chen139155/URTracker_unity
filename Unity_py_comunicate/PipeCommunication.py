import win32pipe, win32file, pywintypes
import threading
import time
import queue
import random

class PipeCommunication:
    def __init__(self, target_pipe, cursor_pipe):
        self.target_pipe = target_pipe
        self.cursor_pipe = cursor_pipe
        self.target_queue = queue.Queue(maxsize=1)  # 只存储最新目标坐标

    def create_pipe(self, pipe_name, access_mode):
        """ 创建命名管道 """
        while True:
            try:
                pipe = win32pipe.CreateNamedPipe(
                    pipe_name,
                    access_mode,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                    1, 65536, 65536, 0, None
                )
                print(f"等待 {pipe_name} 连接...")
                win32pipe.ConnectNamedPipe(pipe, None)
                print(f"{pipe_name} 连接成功！")
                return pipe
            except pywintypes.error as e:
                print(f"创建管道 {pipe_name} 失败，重试中... {e}")
                time.sleep(2)

    def read_target_coordinates(self, unity_message_q):
        """ 读取 Unity 发送的目标坐标 """
        pipe = self.create_pipe(self.target_pipe, win32pipe.PIPE_ACCESS_INBOUND)
        while True:
            try:
                _, data = win32file.ReadFile(pipe, 64)
                target_str = data.decode("utf-8").strip()
                x, y = map(float, target_str.split(","))
                
                # if not self.target_queue.empty():
                #     self.target_queue.get()  # 移除旧数据
                self.unity_message_q.put((x, y))  # 存入最新目标坐标

                print(f"[Python] 目标坐标接收: {x}, {y}")
            except (pywintypes.error, BrokenPipeError):
                print("目标坐标管道断开，尝试重连...")
                time.sleep(2)
                pipe = self.create_pipe(self.target_pipe, win32pipe.PIPE_ACCESS_INBOUND)

    def send_cursor_coordinates(self):
        """ 计算并发送光标坐标 """
        pipe = self.create_pipe(self.cursor_pipe, win32pipe.PIPE_ACCESS_OUTBOUND)
        while True:
            try:
                # 模拟计算 cursor 位置（实际可改为你的逻辑）
                x = random.uniform(-10, 10)
                y = random.uniform(-10, 10)

                if not self.target_queue.empty():
                    target_x, target_y = self.target_queue.queue[0]  # 读取最新目标坐标（不取出）
                    x, y = (target_x + random.uniform(-1, 1), target_y + random.uniform(-1, 1))  # 模拟跟踪目标

                coord_str = f"{x},{y}\n"
                win32file.WriteFile(pipe, coord_str.encode("utf-8"))
                print(f"[Python] 发送光标坐标: {x}, {y}")

                time.sleep(0.016)  # 60Hz 发送速率

            except (pywintypes.error, BrokenPipeError):
                print("光标坐标管道断开，尝试重连...")
                time.sleep(1)
                pipe = self.create_pipe(self.cursor_pipe, win32pipe.PIPE_ACCESS_OUTBOUND)

    def start(self):
        """ 启动目标坐标读取和光标坐标发送的线程 """
        threading.Thread(target=self.read_target_coordinates, daemon=True).start()
        threading.Thread(target=self.send_cursor_coordinates, daemon=True).start()

        # 主线程保持运行
        while True:
            time.sleep(1)

# 创建并启动管道通信
if __name__ == "__main__":
    target_pipe = r'\\.\pipe\UnityToPythonPipe'  # 读取目标坐标
    cursor_pipe = r'\\.\pipe\PythonToUnityPipe'  # 发送光标坐标
    pipe_comm = PipeCommunication(target_pipe, cursor_pipe)
    pipe_comm.start()

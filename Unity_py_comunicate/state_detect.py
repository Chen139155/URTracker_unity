import sys
import torch
import time
import Dynamic_Learning.DL_cc_detect as Dcd

# def compute_state_flag(df_detect, center, W_cog, W_mot1, W_mot2, result_queue):
#     '''计算状态识别结果'''   
#     if 'pygame' in sys.modules:
#         del sys.modules['pygame']
#     time_start = time.time()
#     torch.cuda.empty_cache()  # 清理GPU缓存
#     # 将numpy数组转换为CUDA张量
#     center = torch.from_numpy(center).cuda()
#     W_cog = torch.from_numpy(W_cog).cuda()
#     W_mot1 = torch.from_numpy(W_mot1).cuda()
#     W_mot2 = torch.from_numpy(W_mot2).cuda()
    
#     state_flag, e_cog_norm, e_mot_norm = Dcd.realtime_attention_detect_cc(
#         df_detect, center, W_cog, W_mot1, W_mot2
#     )
#     cognition_result = {'state_flag': state_flag, 'e_cog_norm': e_cog_norm, 'e_mot_norm': e_mot_norm}
#     result_queue.put(cognition_result)
#     print(f'AD Time: {time.time()-time_start:.2f}s')
#     torch.cuda.ipc_collect()  # 补充CUDA IPC资源回收
#     sys.exit(0)  # 显式退出进程


def state_detect_worker(task_queue, result_queue):
    """持续运行的状态检测工作循环"""
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
    
    # 初始化GPU资源（仅执行一次）
    torch.cuda.empty_cache()
    # 生成网格节点
    k = 1       # 修改该参数需要同步修改检测函数中的参数
    eta = 0.2
    print("Start Generating Center Matrix")
    c1 = torch.arange(-k, k + eta, eta)
    c2 = torch.arange(-k, k + eta, eta)
    # c3 = torch.arange(-k, k + eta, eta)

    N = len(c1) * len(c2)
    C1, C2 = torch.meshgrid(c1, c2)

    C1_flat = C1.flatten()
    C2_flat = C2.flatten()
    # C3_flat = C3.flatten()

    center = torch.stack((C1_flat, C2_flat), dim=0)
    print("Center Matrix Generated")
    # 加载权值矩阵
    n = 2 # 修改该参数需要同步修检测函数
    W_cog = torch.zeros(N, n)
    W_mot1 = torch.zeros(N, n)
    W_mot2 = torch.zeros(N, n)

    W_cog = torch.load(r"experiment\urtracker\0225_1147\trained_models\iden_trackball_NOM_A_current.pt")["Wb0"].cuda()
    W_mot1 = torch.load(r"experiment\urtracker\0225_1147\trained_models\iden_trackball_NOM_M_current.pt")["Wb0"].cuda()
    W_mot2 = torch.load(r"experiment\urtracker\0225_1147\trained_models\iden_trackball_ABN_M_current.pt")["Wb0"].cuda()
    
    while True:
        task = task_queue.get()
        if task is None:  # 终止信号
            break
            
        # 解析任务数据
        df_detect, command = task
        
        # 命令处理
        if command == "STOP":
            break
        elif command == "RELOAD_MODEL":
            # 实现模型重载逻辑
            pass
            
        # 执行检测
        try:
            start_time = time.time()
            
            # 转换数据
            center_t = torch.from_numpy(center).cuda()
            W_cog_t = torch.from_numpy(W_cog).cuda()
            W_mot1_t = torch.from_numpy(W_mot1).cuda()
            W_mot2_t = torch.from_numpy(W_mot2).cuda()
            
            # 执行检测
            state_flag, e_cog, e_mot = Dcd.realtime_attention_detect_cc(
                df_detect, center_t, W_cog_t, W_mot1_t, W_mot2_t
            )
            
            # 返回结果
            result_queue.put({
                'status': 'success',
                'data': {
                    'state_flag': state_flag,
                    'e_cog_norm': e_cog,
                    'e_mot_norm': e_mot
                },
                'latency': time.time() - start_time
            })
            
        except Exception as e:
            result_queue.put({
                'status': 'error',
                'message': str(e)
            })
    
    # 清理资源
    torch.cuda.empty_cache()
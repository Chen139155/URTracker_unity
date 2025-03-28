import torch
import data_generator_torch as dg
from RBFNN_lib_torch import RBFNN
from DL_lib_torch import DynamicLearning
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.fft import fft, ifft
import os
import time
# import wandb


# 初始化系统状态、权重、观察器状态和识别错误
def initialize_matrices(n, steps, N):
    xh = torch.zeros((n, steps))
    ek = torch.zeros((n, steps))
    Wb = torch.zeros((N, n))
    # sk = torch.zeros((N, steps))
    return xh, ek, Wb


# 能否加一致性约束？也就是让x和
def calc_max_error(W, xk_normal, center, b, eta):
    '''计算x_hat与输入的xk的最大误差'''
    time1 = time.time()
    ek = torch.zeros([xk_normal.size(0), 1])
    x_hat_list = torch.zeros(xk_normal.shape)
    ek_curve = torch.zeros(xk_normal.shape)
    max_ek = 0.0
    for i in range(xk_normal.size(1) - 1):
        x_hat = W.t().mm(RBFNN.inf(xk_normal[:, i], center, eta))

        # 这个b是干啥的哦
        ek = b * ek + x_hat - xk_normal[:, i + 1].unsqueeze(1)
        ek_curve[:, i] = ek.squeeze()
        x_hat_list[:, i] = x_hat.squeeze()
        if torch.max(torch.abs(ek)) > max_ek:
            max_ek = torch.max(torch.abs(ek))
    print('timeused for calc:', time.time()-time1)
    return max_ek, ek_curve, x_hat_list, xk_normal[:, 1:]

def calc_max_error_cc(W, xk_normal, center, b, eta):
    '''计算x_hat与输入的xk的最大误差'''
    # time1 = time.time()
    n, steps = xk_normal.shape
    ek = torch.zeros(n, 1)
    ek_curve = torch.zeros(n, steps - 1)
    
    for i in range(steps - 1):
        x_hat = W.t().mm(RBFNN.inf(xk_normal[:, i], center, eta))
        ek_curve[:, i] = (b * ek + x_hat - xk_normal[:, i + 1].unsqueeze(1))[:,0]
        # ek_curve[:, i] = ek[:, 0]
    
    ek_norm = torch.mean(ek_curve[:, 10:-10] ** 2)
    # print('timeused for calc:', time.time() - time1)
    return ek_norm

# def calc_max_error_cc(W, xk_normal, center, b, eta):
#     '''计算x_hat与输入的xk的最大误差'''
    
#     # 记录开始时间
#     start_time = time.time()
    
#     n, steps = xk_normal.shape
#     ek = torch.zeros(n, 1)
#     ek_curve = torch.zeros(n, steps - 1)
    
#     # 计算每个时间步的 x_hat 和 ek
#     for i in range(steps - 1):
#         # 记录开始时间
#         start_step_time = time.time()
        
#         # 计算 RBFNN.inf 的时间
#         start_rbfnn_time = time.time()
#         rbfnn_output = RBFNN.inf(xk_normal[:, i], center, eta)
#         end_rbfnn_time = time.time()
#         print(f"RBFNN.inf took {end_rbfnn_time - start_rbfnn_time:.4f} seconds")
        
#         # 记录 W.t().mm(RBFNN) 的时间
#         start_mm_time = time.time()
#         x_hat = W.t().mm(rbfnn_output)
#         end_mm_time = time.time()
#         print(f"W.t().mm(RBFNN) took {end_mm_time - start_mm_time:.4f} seconds")
        
#         ek = b * ek + x_hat - xk_normal[:, i + 1].unsqueeze(1)
#         ek_curve[:, i] = ek[:, 0]
        
#         # 记录结束时间并打印时间戳
#         end_step_time = time.time()
#         print(f"Step {i} took {end_step_time - start_step_time:.4f} seconds")
    
#     # 计算 ek_curve 的均方误差
#     start_mean_time = time.time()
#     ek_norm = torch.mean(ek_curve[:, 10:-10] ** 2)
#     end_mean_time = time.time()
#     print(f"Mean calculation took {end_mean_time - start_mean_time:.4f} seconds")
    
#     # 记录总结束时间
#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time:.4f} seconds")
    
#     return ek_norm


def low_pass_filter(signal, cutoff_frequency, sampling_rate):
    """
    使用FFT实现低通滤波器。
    
    参数:
    - signal: 输入信号（时间序列）
    - cutoff_frequency: 截止频率
    - sampling_rate: 采样率
    
    返回值:
    - filtered_signal: 经过低通滤波的信号
    """
    # 计算FFT
    fft_signal = fft(signal)
    frequencies = torch.fft.fftfreq(signal.size(0), d=1 / sampling_rate)

    # 构造低通滤波器
    cutoff_idx = torch.abs(frequencies) > cutoff_frequency
    fft_signal[cutoff_idx] = 0

    # 计算反FFT
    filtered_signal = ifft(fft_signal)

    return filtered_signal.real


def push_to_tensor_alternative(tensor, x, length=50):
    '''固定长度的滑动窗口或者队列，新的元素不断加入，旧的元素在队列满时被移除'''
    if tensor is None:
        return x.unsqueeze(0)
    if tensor.size(0) < length:
        return torch.cat((tensor, x.unsqueeze(0)))
    else:
        return torch.cat((tensor[1:], x.unsqueeze(0)))


def start_dynamic_learning(xk_train, xk_test, N, n, center, b, eta, Ao, epoch, gamma, avg_length, outf, mode, time_tag, resume):
    model_outf = "experiment/{}/{}/trained_models/".format(outf, time_tag)
    curve_outf = "experiment/{}/{}/ek_curves/".format(outf, time_tag)

    if not os.path.exists(model_outf): os.makedirs(model_outf, exist_ok=True)
    if not os.path.exists(curve_outf): os.makedirs(curve_outf, exist_ok=True)
    Wb0 = torch.zeros(N, n)
    if resume != "":
        Wb0 = torch.load(resume)
    Wb0_list = None
    Wb0_flat_history = torch.zeros(epoch, N * n)
    Wb0_avg_flat_history = torch.zeros(epoch, N * n)

    x0_train = xk_train[:, 0].reshape(-1, 1)
    train_steps = xk_train.size(1)
    list_em0_train = []
    list_em0_val = []

    will_train = True
    will_test = True

    for i in range(epoch):
        # train stage
        if will_train:
            xh0, ek0, Wb0 = DynamicLearning.learn_mini(
                x0_train, xk_train, Wb0, train_steps, center, eta, Ao, gamma
            )
            Wb0_flat_history[i] = Wb0.flatten()

            train_em0, ek_curve, x_hat_list, _ = calc_max_error(Wb0, xk_train, center, b, eta)

            Wb0_list = push_to_tensor_alternative(Wb0_list, Wb0, avg_length)

            # plt.imshow(Wb0.reshape([len(c1), len(c2), len(c3), len(c4), 4]).sum(axis=2).sum(axis=2)[:, :, :2].sum(axis=2).cpu().numpy())

            plt.figure()
            plt.title("ek_curve of epoch {}".format(i))
            plt.plot(ek_curve[0].cpu(), label='x')
            plt.plot(ek_curve[1].cpu(), label='y')
            if len(ek_curve) > 2:
                plt.plot(ek_curve[2].cpu(), label='vx')
                plt.plot(ek_curve[3].cpu(), label='vy')
            plt.ylim(-0.25, 0.25)
            plt.legend()

            plt.savefig(curve_outf + "ek_curve_epoch_{}.png".format(i))
            plt.close()
            # plt.show()

            print("mean train ek0: ", ek0.mean(dim=1), "val_em0: ", train_em0)
            # wandb.log({"mean train ek0": ek0.mean(), "val_em0": train_em0})
            list_em0_train.append(train_em0.item())

            avg_Wb0 = torch.mean(Wb0_list, dim=0)
            Wb0_avg_flat_history[i] = avg_Wb0.flatten()

            torch.save(
                {
                    "Wb0": Wb0,
                    "avg_Wb0": avg_Wb0,
                    "ek0": ek0,
                    "train_em0": train_em0,
                    "xk_normal": xk_train,
                },
                model_outf + "iden_trackball_{}_current.pt".format(mode),
            )

            plt.figure()
            plt.title("visible Wb0 of epoch {}".format(i))
            plt.plot(Wb0_flat_history.cpu().numpy())
            plt.savefig(curve_outf + "Wb0_curve_epoch_{}.png".format(i))
            plt.xlim(0, epoch)
            plt.close()

        if will_test:
            val_em0, ek_curve, x_hat_list, _ = calc_max_error(avg_Wb0, xk_test, center, b, eta)

            plt.figure()
            plt.title("val ek_curve of epoch {}".format(i))
            plt.plot(ek_curve[0].cpu(), label='x')
            plt.plot(ek_curve[1].cpu(), label='y')
            if len(ek_curve) > 2:
                plt.plot(ek_curve[2].cpu(), label='vx')
                plt.plot(ek_curve[3].cpu(), label='vy')
            plt.ylim(-0.25, 0.25)
            plt.legend()

            plt.savefig(curve_outf + "val_ek_curve_epoch_{}.png".format(i))
            plt.close()
            # plt.show()

            print("mean val ek0: ", ek0.mean(dim=1), "val_em0: ", val_em0)
            # wandb.log({"mean val ek0": ek0.mean(), "val_em0": val_em0})
            list_em0_val.append(val_em0.item())

    plt.figure()
    plt.plot(list_em0_train)
    plt.savefig(curve_outf + "train_ek_curve_epoch_total.png")
    plt.figure()
    plt.plot(list_em0_val)
    plt.savefig(curve_outf + "val_ek_curve_epoch_total.png")

    print("Finished {} Epoches Training".format(epoch))
    return Wb0, Wb0_list, avg_Wb0


def start_dynamic_learning_old(x0_normal, xk_normal, N, n, steps, center, b, eta, Ao, epoch, avg_length, outf, mode, time_tag, resume):
    """
        动态学习训练参数 w

        :param x0_norma: 
        :param xk_normal:
        :param N:  神经元结点数量
        :param n:  RBFNN 维度
        
        :return:
        return 
    """
    model_outf = "experiment/{}/{}/trained_models/".format(outf, time_tag)
    curve_outf = "experiment/{}/{}/ek_curves/".format(outf, time_tag)

    if not os.path.exists(model_outf): os.makedirs(model_outf, exist_ok=True)
    if not os.path.exists(curve_outf): os.makedirs(curve_outf, exist_ok=True)
    print('xk_normal start size :', xk_normal.shape)
    Wb0 = torch.zeros(N, n)
    if resume != "":
        Wb0 = torch.load(resume)
    Wb0_list = None
    list_em0 = []
    for i in range(epoch):
        xh0, ek0, Wb0 = DynamicLearning.learn_mini(
            x0_normal, xk_normal, Wb0, steps, center, eta, Ao, gamma=0.04
        )
        # print('xk_normal size :', xk_normal.shape)
        val_em0, ek_curve, x_hat_list, _ = calc_max_error(Wb0, xk_normal, center, b, eta)

        Wb0_list = push_to_tensor_alternative(Wb0_list, Wb0, avg_length)

        # plt.imshow(Wb0.reshape([len(c1), len(c2), len(c3), len(c4), 4]).sum(axis=2).sum(axis=2)[:, :, :2].sum(axis=2).cpu().numpy())

        # plt.figure()
        # plt.title("ek_curve of epoch {}".format(i))
        # plt.plot(ek_curve[0,100:-100].cpu(), label='RT_r')
        # plt.plot(ek_curve[1,100:-100].cpu(), label='TE')
        # plt.plot(ek_curve[2,100:-100].cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        # plt.legend()

        # plt.savefig(curve_outf + "ek_curve_epoch_{}_{}.png".format(i,mode))
        # plt.close()
        # plt.show()

        print("epoch: ", i, "mean train ek0: ", ek0.mean(dim=1), "val_em0: ", val_em0)
        # wandb.log({"mean train ek0": ek0.mean(), "val_em0": val_em0})
        list_em0.append(ek0.mean().item())

        avg_Wb0 = torch.mean(Wb0_list, dim=0)

        torch.save(
            {
                "Wb0": Wb0,
                "avg_Wb0": avg_Wb0,
                "ek0": ek0,
                "val_em0": val_em0,
                "xk_normal": xk_normal,
                "x0_normal": x0_normal,
            },
            model_outf + "iden_trackball_{}_current.pt".format(mode),
        )

    plt.figure()
    plt.plot(list_em0)
    plt.savefig(curve_outf + "ek_curve_epoch_total_{}.png".format(mode))

    print("Finished {} Epoches Training".format(epoch))
    return Wb0, Wb0_list, avg_Wb0

def start_dynamic_learning_cc(x0_normal, xk_normal, N, n, steps, center, b, eta, Ao, epoch, avg_length, outf, mode, time_tag, resume):
    """
        动态学习训练参数 w

        :param x0_norma: 
        :param xk_normal:
        :param N:  神经元结点数量
        :param n:  RBFNN 维度
        
        :return:
        return 
    """
    model_outf = "experiment/{}/{}/trained_models/".format(outf, time_tag)
    curve_outf = "experiment/{}/{}/ek_curves/".format(outf, time_tag)

    if not os.path.exists(model_outf): os.makedirs(model_outf, exist_ok=True)
    if not os.path.exists(curve_outf): os.makedirs(curve_outf, exist_ok=True)
    print('xk_normal start size :', xk_normal.shape)
    Wb0 = torch.zeros(N, n)
    if resume != "":
        Wb0 = torch.load(resume)
    Wb0_list = None
    list_em0 = []
    for i in range(epoch):
        xh0, ek0, Wb0 = DynamicLearning.learn_mini(
            x0_normal, xk_normal, Wb0, steps, center, eta, Ao, gamma=0.04
        )
        # print('xk_normal size :', xk_normal.shape)
        val_em0, ek_curve, x_hat_list, _ = calc_max_error(Wb0, xk_normal, center, b, eta)

        Wb0_list = push_to_tensor_alternative(Wb0_list, Wb0, avg_length)

        # plt.imshow(Wb0.reshape([len(c1), len(c2), len(c3), len(c4), 4]).sum(axis=2).sum(axis=2)[:, :, :2].sum(axis=2).cpu().numpy())

        # plt.figure()
        # plt.title("ek_curve of epoch {}".format(i))
        # plt.plot(ek_curve[0,100:-100].cpu(), label='RT_r')
        # plt.plot(ek_curve[1,100:-100].cpu(), label='TE')
        # plt.plot(ek_curve[2,100:-100].cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        # plt.legend()

        # plt.savefig(curve_outf + "ek_curve_epoch_{}_{}.png".format(i,mode))
        # plt.close()
        # plt.show()

        print("epoch: ", i, "mean train ek0: ", ek0.mean(dim=1), "val_em0: ", val_em0)
        # wandb.log({"mean train ek0": ek0.mean(), "val_em0": val_em0})
        list_em0.append(ek0.mean().item())

        avg_Wb0 = torch.mean(Wb0_list, dim=0)

        torch.save(
            {
                "Wb0": Wb0,
                "avg_Wb0": avg_Wb0,
                "ek0": ek0,
                "val_em0": val_em0,
                "xk_normal": xk_normal,
                "x0_normal": x0_normal,
                "xh0": xh0,
            },
            model_outf + "iden_trackball_{}_current.pt".format(mode),
        )

    plt.figure()
    plt.plot(list_em0)
    plt.savefig(curve_outf + "ek_curve_epoch_total_{}.png".format(mode))

    print("Finished {} Epoches Training".format(epoch))
    return Wb0, Wb0_list, avg_Wb0
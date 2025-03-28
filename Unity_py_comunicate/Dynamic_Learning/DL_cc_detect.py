"""
Function:
    动态模式识别的detect模块
Author:
    MuFengjun, CC 
"""
import torch
import data_generator_torch as dg
import DL_DataProcess as DD
import DL_cc_DataProcess as DcD
from RBFNN_lib_torch import RBFNN
from DL_lib_torch import DynamicLearning
import matplotlib.pyplot as plt
import torch.nn.functional as F
import util
# import wandb
import time
import argparse
import os
import pandas as pd

def visuial_attention_detect_test():
    '''视觉注意力abnormaity检测的离线测试函数'''
    # 初始化网络参数
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    this_time = "test" + str(time.time())

    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=1.0)
    parser.add_argument('--eta', default=0.1)
    parser.add_argument('--b', default=0.2) # 
    parser.add_argument('--k', default=1.0) # RBFNN边界范围
    parser.add_argument('--epoch', default=500)
    parser.add_argument('--Ao', default=0.1)
    parser.add_argument('--output_dir', type=str, default='tobii_ori')
    parser.add_argument('--avg_length', default=5)

    opt = parser.parse_args()

    k = opt.k
    eta = opt.eta
    gamma = opt.gamma
    b = opt.b
    epoch = opt.epoch
    Ao = opt.Ao * torch.eye(3)

    # 生成网格节点
    print("Start Generating Center Matrix")
    c1 = torch.arange(-k, k + eta, eta)
    c2 = torch.arange(-k, k + eta, eta)
    c3 = torch.arange(-k, k + eta, eta)

    N = len(c1) * len(c2) * len(c3)
    C1, C2, C3 = torch.meshgrid(c1, c2, c3)

    C1_flat = C1.flatten()
    C2_flat = C2.flatten()
    C3_flat = C3.flatten()

    center = torch.stack((C1_flat, C2_flat, C3_flat), dim=0)
    print("Center Matrix Generated")
    # 导入数据
    print("Start Loading Data")
    type = "sin"
    reprocess_data = False
    if reprocess_data:
        xk_trackball_focus = DD.get_urtracker_data(
            path="output/1736423776.7707956mode1_test01.xlsx".format(type)
        )
        xk_realtime = DD.get_urtracker_data(
            path="output/1736423604.7432425mode2_test01.xlsx".format(type)
        )
        xk_trackball_unfocus2 = DD.get_urtracker_data(
            path="output/1736423305.6951458mode3_test01.xlsx".format(type)
        )
        xk_trackball_unfocus3 = DD.get_urtracker_data(
            path="output/1736423137.817998mode4_test01.xlsx".format(type)
        )

        xk_normal = torch.cat(
            [
                xk_trackball_focus["RT_gaze"].unsqueeze(0),
                xk_trackball_focus["TE_gaze"].unsqueeze(0),
                xk_trackball_focus["PI_gaze"].unsqueeze(0),
            ],
            dim=0,
        )
        print("xk_normal shape:", xk_normal.shape)

        xk_fault1 = torch.cat(
            [
                xk_realtime["RT_gaze"].unsqueeze(0),
                xk_realtime["TE_gaze"].unsqueeze(0),
                xk_realtime["PI_gaze"].unsqueeze(0),
            ]
        )

        print("xk_fault shape:", xk_fault1.shape)

        xk_fault2 = torch.cat(
            [
                xk_trackball_unfocus2["RT_gaze"].unsqueeze(0),
                xk_trackball_unfocus2["TE_gaze"].unsqueeze(0),
                xk_trackball_unfocus2["PI_gaze"].unsqueeze(0),
            ]
        )

        print("xk_fault2 shape:", xk_fault2.shape)

        xk_fault3 = torch.cat(
            [
                xk_trackball_unfocus3["RT_gaze"].unsqueeze(0),
                xk_trackball_unfocus3["TE_gaze"].unsqueeze(0),
                xk_trackball_unfocus3["PI_gaze"].unsqueeze(0),
            ]
        )

        print("xk_fault3 shape:", xk_fault3.shape)

        # 绘图debug
        draw_debug = True

        if draw_debug:
            # plt.close()
            show_length = 1300
            plt.figure("x curve")
            plt.plot(xk_normal[0, :show_length].cpu(), label="normal")
            plt.plot(xk_fault1[0, :show_length].cpu(), label="fault1")
            plt.plot(xk_fault2[0, :show_length].cpu(), label="fault2")
            plt.plot(xk_fault3[0, :show_length].cpu(), label="fault3")
            plt.legend()
            plt.show()

            plt.figure("y curve")
            plt.plot(xk_normal[1, :show_length].cpu(), label="normal")
            plt.plot(xk_fault1[1, :show_length].cpu(), label="fault1")
            plt.plot(xk_fault2[1, :show_length].cpu(), label="fault2")
            plt.plot(xk_fault3[1, :show_length].cpu(), label="fault3")
            plt.legend()
            plt.show()

            plt.figure("vx curve")
            plt.plot(xk_normal[2, :show_length].cpu(), label="normal")
            plt.plot(xk_fault1[2, :show_length].cpu(), label="fault1")
            plt.plot(xk_fault2[2, :show_length].cpu(), label="fault2")
            plt.plot(xk_fault3[2, :show_length].cpu(), label="fault3")
            plt.legend()
            plt.show()

            # plt.figure("vy curve")
            # plt.plot(xk_normal[3, :show_length].cpu(), label="normal")
            # plt.plot(xk_fault1[3, :show_length].cpu(), label="fault1")
            # plt.plot(xk_fault2[3, :show_length].cpu(), label="fault2")
            # plt.plot(xk_fault3[3, :show_length].cpu(), label="fault3")
            # plt.legend()
            # plt.show()

        torch.save(xk_normal, "output/xk_normal_robio_{}.pt".format(type))
        torch.save(xk_fault1, "output/xk_fault1_robio_{}.pt".format(type))
        torch.save(xk_fault2, "output/xk_fault2_robio_{}.pt".format(type))
        torch.save(xk_fault3, "output/xk_fault3_robio_{}.pt".format(type))
    else:
        xk_normal = torch.load("output/xk_normal_robio_{}.pt".format(type))
        xk_fault1 = torch.load("output/xk_fault1_robio_{}.pt".format(type))
        xk_fault2 = torch.load("output/xk_fault2_robio_{}.pt".format(type))
        xk_fault3 = torch.load("output/xk_fault3_robio_{}.pt".format(type))

    factors = torch.tensor([[0.03, 0.04, 0.4]], device='cuda').repeat(910, 1).T
    vec_factor = 0.1
    xk_normal = xk_normal[:, 480:1390]
    xk_fault1 = xk_fault1[:, 480:1390]
    xk_fault2 = xk_fault2[:, 480:1390]
    xk_fault3 = xk_fault3[:, 480:1390]
    
    xk_normal = k * F.tanh(factors * xk_normal)
    xk_fault1 = k * F.tanh(factors * xk_fault1)
    xk_fault2 = k * F.tanh(factors * xk_fault2)
    xk_fault3 = k * F.tanh(factors * xk_fault3)

    print("xk_normal shape:", xk_normal.shape)
    x0_normal = xk_normal[:, 0].reshape(-1, 1)  # 列向量
    x0_fault1 = xk_fault1[:, 0].reshape(-1, 1)
    x0_fault2 = xk_fault2[:, 0].reshape(-1, 1)
    x0_fault3 = xk_fault3[:, 0].reshape(-1, 1)

    n = xk_normal.size(0)
    steps = xk_normal.size(1)
    print("Data Loaded")

    # Start Detecting
    print("Start Detecting")
    Wb0 = torch.zeros(N, n)

    Wb0 = torch.load("experiment/urtracker/1739854132.5614216/trained_models/iden_trackball_m2_normal_current.pt")["Wb0"].cuda()
    outf = "experiment/urtracker/{}/".format(this_time)
    if not os.path.exists(outf): os.makedirs(outf, exist_ok=True)

    val_em0, ek_curve0, x_hat0, x_k0 = util.calc_max_error(Wb0, xk_normal, center, b, eta)
    val_em1, ek_curve1, x_hat1, x_k1 = util.calc_max_error(Wb0, xk_fault1, center, b, eta)
    val_em2, ek_curve2, x_hat2, x_k2 = util.calc_max_error(Wb0, xk_fault2, center, b, eta)
    val_em3, ek_curve3, x_hat3, x_k3 = util.calc_max_error(Wb0, xk_fault3, center, b, eta)

    curout_frequency = 0.1
    sampling_rate = 50

    plt.figure()
    plt.title("Normal Mode")
    plt.plot(ek_curve0[0].cpu(), label='RT_r')
    plt.plot(ek_curve0[1].cpu(), label='TE')
    plt.plot(ek_curve0[2].cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "normal.png")
    plt.show()

    max_normal = torch.max(torch.cat([util.low_pass_filter(ek_curve0[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve0[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve0[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    max_x = torch.argmax(max_normal)
    max_y = torch.max(max_normal[max_x])
    plt.figure()
    plt.title("Normal Error Curve")
    plt.plot(max_normal)
    plt.ylim(-0.9, 0.9)
    plt.scatter(max_x, max_y, color='red', s=50)
    plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.savefig(outf + "normal_error_curve.png")
    plt.show()

    plt.figure()
    plt.title("Normal Mode Filter")
    plt.plot(util.low_pass_filter(ek_curve0[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    plt.plot(util.low_pass_filter(ek_curve0[1], curout_frequency, sampling_rate).cpu(), label='TE')
    plt.plot(util.low_pass_filter(ek_curve0[2], curout_frequency, sampling_rate).cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "normal_filter.png")
    plt.show()

    plt.figure()
    plt.title("Abnormal Mode I")
    plt.plot(ek_curve1[0].cpu(), label='RT_r')
    plt.plot(ek_curve1[1].cpu(), label='TE')
    plt.plot(ek_curve1[2].cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault1.png")
    plt.show()

    plt.figure()
    plt.title("Abnormal Mode I Filter")
    plt.plot(util.low_pass_filter(ek_curve1[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    plt.plot(util.low_pass_filter(ek_curve1[1], curout_frequency, sampling_rate).cpu(), label='TE')
    plt.plot(util.low_pass_filter(ek_curve1[2], curout_frequency, sampling_rate).cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault1_filter.png")
    plt.show()

    max_fault1 = torch.max(torch.cat([util.low_pass_filter(ek_curve1[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve1[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve1[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    max_x = torch.argmax(max_fault1)
    max_y = torch.max(max_fault1[max_x])
    # print("early detect of fault1:", max_fault1.gt(0).nonzero(as_tuple=True)[0][0])
    plt.figure()
    plt.title("Abnormal Mode I Error Curve")
    plt.plot(max_fault1)
    plt.ylim(-0.9, 0.9)
    plt.scatter(max_x, max_y, color='red', s=50)
    plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x + 2000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    print(max_fault1.gt(0.0114).nonzero(as_tuple=True))
    plt.savefig(outf + "fault1_error_curve.png")
    plt.show()

    plt.figure()
    plt.title("Abnormal Mode II")
    plt.plot(ek_curve2[0].cpu(), label='RT_r')
    plt.plot(ek_curve2[1].cpu(), label='TE')
    plt.plot(ek_curve2[2].cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault2.png")

    plt.show()

    plt.figure()
    plt.title("Abnormal Mode II Filter")
    plt.plot(util.low_pass_filter(ek_curve2[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    plt.plot(util.low_pass_filter(ek_curve2[1], curout_frequency, sampling_rate).cpu(), label='TE')
    plt.plot(util.low_pass_filter(ek_curve2[2], curout_frequency, sampling_rate).cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault2_filter.png")
    plt.show()

    max_fault2 = torch.max(torch.cat([util.low_pass_filter(ek_curve2[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve2[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve2[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    # print("early detect of fault2:", max_fault2.gt(val_em0.item()).nonzero(as_tuple=True)[0][0])

    max_x = torch.argmax(max_fault2)
    max_y = torch.max(max_fault2[max_x])

    plt.figure()
    plt.title("Abnormal Mode II Error Curve")
    plt.plot(max_fault2)
    plt.ylim(-0.9, 0.9)
    plt.scatter(max_x, max_y, color='red', s=50)
    plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.savefig(outf + "fault2_error_curve.png")
    plt.show()

    plt.figure()
    plt.title("Abnormal Mode III")
    plt.plot(ek_curve3[0].cpu(), label='RT_r')
    plt.plot(ek_curve3[1].cpu(), label='TE')
    plt.plot(ek_curve3[2].cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault3.png")

    plt.show()

    plt.figure()
    plt.title("Abnormal Mode III Filter")
    plt.plot(util.low_pass_filter(ek_curve3[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    plt.plot(util.low_pass_filter(ek_curve3[1], curout_frequency, sampling_rate).cpu(), label='TE')
    plt.plot(util.low_pass_filter(ek_curve3[2], curout_frequency, sampling_rate).cpu(), label='PI')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "fault3_filter.png")

    plt.show()

    max_fault3 = torch.max(torch.cat([util.low_pass_filter(ek_curve3[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve3[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                    util.low_pass_filter(ek_curve3[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    # print("early detect of fault3:", max_fault3.gt(val_em0.item()).nonzero(as_tuple=True)[0][0])

    max_x = torch.argmax(max_fault3)
    max_y = torch.max(max_fault3[max_x])

    plt.figure()
    plt.title("Abnormal Mode III Error Curve")
    plt.plot(max_fault3)
    plt.ylim(-0.9, 0.9)
    plt.scatter(max_x, max_y, color='red', s=50)
    plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x + 2000, max_y - 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    plt.savefig(outf + "fault3_error_curve.png")
    plt.show()

    plt.figure()
    plt.title("Comparison of Error Curves with W1")
    plt.plot(max_normal, label='Normal Mode')
    plt.plot(max_fault1, label='Abnormal Mode I')
    plt.plot(max_fault2, label='Abnormal Mode II')
    plt.plot(max_fault3, label='Abnormal Mode III')
    plt.ylim(-0.9, 0.9)
    plt.legend()
    plt.savefig(outf + "comp_error_curve.png")

def realtime_attention_detect(data_path, draw_debug=False):
    # 初始化网络参数
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    this_time = "Detect" + time.strftime("%m%d_%H%M")

    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=1.0)
    parser.add_argument('--eta', default=0.1)
    parser.add_argument('--b', default=0.2) # 
    parser.add_argument('--k', default=1.0) # RBFNN边界范围
    parser.add_argument('--epoch', default=500)
    parser.add_argument('--Ao', default=0.1)
    parser.add_argument('--output_dir', type=str, default='tobii_ori')
    parser.add_argument('--avg_length', default=5)

    opt = parser.parse_args()

    k = opt.k
    eta = opt.eta
    gamma = opt.gamma
    b = opt.b
    epoch = opt.epoch
    Ao = opt.Ao * torch.eye(3)

    # 生成网格节点
    print("Start Generating Center Matrix")
    c1 = torch.arange(-k, k + eta, eta)
    c2 = torch.arange(-k, k + eta, eta)
    c3 = torch.arange(-k, k + eta, eta)

    N = len(c1) * len(c2) * len(c3)
    C1, C2, C3 = torch.meshgrid(c1, c2, c3)

    C1_flat = C1.flatten()
    C2_flat = C2.flatten()
    C3_flat = C3.flatten()

    center = torch.stack((C1_flat, C2_flat, C3_flat), dim=0)
    print("Center Matrix Generated")
    # 导入数据
    print("Start Loading Data")
    type = "sin"
    reprocess_data = True
    if reprocess_data:
        xk_realtime = DD.get_urtracker_data(
            path = data_path
        )
        # 实时眼动数据
        xk_gaze = torch.cat(
            [
                xk_realtime["entire_raw_gaze_x"].unsqueeze(0),
                xk_realtime["entire_raw_gaze_y"].unsqueeze(0),
                xk_realtime["PI_gaze"].unsqueeze(0),
            ],
            dim=0,
        )
        print("xk_gaze shape:", xk_gaze.shape)
        # 实时运动数据
        xk_motion = torch.cat(
            [
                xk_realtime["entire_raw_x"].unsqueeze(0),
                xk_realtime["entire_raw_y"].unsqueeze(0),
                xk_realtime["force_norm"].unsqueeze(0),
            ]
        )
        print("xk_motion shape:", xk_motion.shape)



        

        torch.save(xk_gaze, "output/xk_gaze_realtime_{}.pt".format(type))
        torch.save(xk_motion, "output/xk_motion_realtime{}.pt".format(type))
    else:
        xk_gaze = torch.load("output/xk_gaze_realtime_{}.pt".format(type))
        xk_motion = torch.load("output/xk_motion_realtime_{}.pt".format(type))

    xk_gaze[2,:] = xk_gaze[2,:]-4.5
    data_length = xk_gaze.shape[1]
    gaze_factors = torch.tensor([[0.004, 0.004, 0.02]], device='cuda').repeat(data_length, 1).T
    motion_factors = torch.tensor([[0.004, 0.004, 0.02]], device='cuda').repeat(data_length, 1).T
    # factors = torch.tensor([[0.03, 0.04, 0.4]], device='cuda').repeat(data_length, 1).T
    # vec_factor = 0.1
    
    xk_gaze = k * F.tanh(gaze_factors * xk_gaze)
    xk_motion = k * F.tanh(motion_factors * xk_motion)

    # 绘图debug
        # draw_debug = False

    if draw_debug:
        # plt.close()
        show_length = 4000
        plt.figure("RT reatime")
        plt.plot(xk_gaze[0, :show_length].cpu(), label="gaze_reatime")
        plt.plot(xk_motion[0, :show_length].cpu(), label="motion_reatime")
        plt.legend()
        plt.show()

        plt.figure("TE realtime")
        plt.plot(xk_gaze[1, :show_length].cpu(), label="gaze")
        plt.plot(xk_motion[1, :show_length].cpu(), label="motion")
        plt.legend()
        plt.show()

        plt.figure("....realtime")
        plt.plot(xk_gaze[2, :show_length].cpu(), label="gaze")
        plt.plot(xk_motion[2, :show_length].cpu(), label="motion")
        plt.legend()
        plt.show()

    print("xk_gaze shape:", xk_gaze.shape)
    x0_normal = xk_gaze[:, 0].reshape(-1, 1)  # 列向量
    x0_fault1 = xk_motion[:, 0].reshape(-1, 1)

    n = xk_gaze.size(0)
    steps = xk_gaze.size(1)
    print("Data Loaded")

    # Start Detecting
    print("Start Detecting")
    W_cog = torch.zeros(N, n)
    W_mot1 = torch.zeros(N, n)
    W_mot2 = torch.zeros(N, n)

    W_cog = torch.load("experiment/urtracker/0219_2304/trained_models/iden_trackball_NOM_A_current.pt")["Wb0"].cuda()
    W_mot1 = torch.load("experiment/urtracker/0219_2304/trained_models/iden_trackball_NOM_M_current.pt")["Wb0"].cuda()
    W_mot2 = torch.load("experiment/urtracker/0219_2304/trained_models/iden_trackball_ABN_M_current.pt")["Wb0"].cuda()
    outf = "experiment/urtracker/{}/".format(this_time)
    if not os.path.exists(outf): os.makedirs(outf, exist_ok=True)

    val_em0, ek_curve_cog, x_hat0, x_k0 = util.calc_max_error(W_cog, xk_gaze, center, b, eta)
    val_em1, ek_curve_mot1, x_hat1, x_k1 = util.calc_max_error(W_mot1, xk_motion, center, b, eta)
    val_em2, ek_curve_mot2, x_hat2, x_k2 = util.calc_max_error(W_mot2, xk_motion, center, b, eta)
    # val_em3, ek_curve3, x_hat3, x_k3 = util.calc_max_error(Wb0, xk_fault3, center, b, eta)
    e_cog = torch.mean(ek_curve_cog ** 2)
    e_mot1 = torch.mean(ek_curve_mot1 ** 2)
    e_mot2 = torch.mean(ek_curve_mot2 ** 2)
    print('e_cog, e_mot1, e_mot2: ', e_cog, e_mot1, e_mot2)
    curout_frequency = 0.1
    sampling_rate = 20
    # Drawfig = True
    if draw_debug:
        '''w_cog模型结果'''
        # 绘制ek_curve_cog
        plt.figure()
        plt.title("ek_curve_cog")
        plt.plot(ek_curve_cog[0].cpu(), label='ek_RT_gaze')
        plt.plot(ek_curve_cog[1].cpu(), label='ek_TE_gaze')
        plt.plot(ek_curve_cog[2].cpu(), label='ek_PI_gaze')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_curve_cog.png")
        plt.show()
        # 绘制ek_curvce三维最大值
        max_normal = torch.max(torch.cat([util.low_pass_filter(ek_curve_cog[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_cog[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_cog[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

        max_x = torch.argmax(max_normal)
        max_y = torch.max(max_normal[max_x])
        plt.figure()
        plt.title("Max Error Curve cog")
        plt.plot(max_normal)
        # plt.ylim(-0.9, 0.9)
        plt.scatter(max_x, max_y, color='red', s=50)
        plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
        plt.savefig(outf + "max_error_curve_cog.png")
        plt.show()

        plt.figure()
        plt.title("ek_curve_cog Filter")
        plt.plot(util.low_pass_filter(ek_curve_cog[0], curout_frequency, sampling_rate).cpu(), label='RT_gaze')
        plt.plot(util.low_pass_filter(ek_curve_cog[1], curout_frequency, sampling_rate).cpu(), label='TE_gaze')
        plt.plot(util.low_pass_filter(ek_curve_cog[2], curout_frequency, sampling_rate).cpu(), label='PI_gaze')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_cog_filter.png")
        plt.show()
        ''' w_mot1 输出结果'''
        # 绘制ek_curve_mot1
        plt.figure()
        plt.title("ek_curve_mot1")
        plt.plot(ek_curve_mot1[0].cpu(), label='RT')
        plt.plot(ek_curve_mot1[1].cpu(), label='TE')
        plt.plot(ek_curve_mot1[2].cpu(), label='force')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_curve_mot1.png")
        plt.show()

        plt.figure()
        plt.title("ek_curve_mot1_filter")
        plt.plot(util.low_pass_filter(ek_curve_mot1[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
        plt.plot(util.low_pass_filter(ek_curve_mot1[1], curout_frequency, sampling_rate).cpu(), label='TE')
        plt.plot(util.low_pass_filter(ek_curve_mot1[2], curout_frequency, sampling_rate).cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_curve_mot1_filter.png")
        plt.show()
        # 绘制最大值
        max_fault1 = torch.max(torch.cat([util.low_pass_filter(ek_curve_mot1[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_mot1[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_mot1[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

        max_x = torch.argmax(max_fault1)
        max_y = torch.max(max_fault1[max_x])
        # print("early detect of fault1:", max_fault1.gt(0).nonzero(as_tuple=True)[0][0])
        plt.figure()
        plt.title("Abnormal Mode I Error Curve")
        plt.plot(max_fault1)
        # plt.ylim(-0.9, 0.9)
        plt.scatter(max_x, max_y, color='red', s=50)
        plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x + 2000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
        print(max_fault1.gt(0.0114).nonzero(as_tuple=True))
        plt.savefig(outf + "fault1_error_curve.png")
        plt.show()
        '''mot2'''
        plt.figure()
        plt.title("ek_curve_mot2")
        plt.plot(ek_curve_mot2[0].cpu(), label='RT_r')
        plt.plot(ek_curve_mot2[1].cpu(), label='TE')
        plt.plot(ek_curve_mot2[2].cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_curve_mot2.png")

        plt.show()

        plt.figure()
        plt.title("ek_curve_mot2_filter")
        plt.plot(util.low_pass_filter(ek_curve_mot2[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
        plt.plot(util.low_pass_filter(ek_curve_mot2[1], curout_frequency, sampling_rate).cpu(), label='TE')
        plt.plot(util.low_pass_filter(ek_curve_mot2[2], curout_frequency, sampling_rate).cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "ek_curve_mot2_filter.png")
        plt.show()

        max_fault2 = torch.max(torch.cat([util.low_pass_filter(ek_curve_mot2[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_mot2[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
                                        util.low_pass_filter(ek_curve_mot2[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

        # print("early detect of fault2:", max_fault2.gt(val_em0.item()).nonzero(as_tuple=True)[0][0])

        max_x = torch.argmax(max_fault2)
        max_y = torch.max(max_fault2[max_x])

        plt.figure()
        plt.title("Abnormal Mode II Error Curve")
        plt.plot(max_fault2)
        # plt.ylim(-0.9, 0.9)
        plt.scatter(max_x, max_y, color='red', s=50)
        plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
        plt.savefig(outf + "fault2_error_curve.png")
        plt.show()

        # plt.figure()
        # plt.title("Abnormal Mode III")
        # plt.plot(ek_curve3[0].cpu(), label='RT_r')
        # plt.plot(ek_curve3[1].cpu(), label='TE')
        # plt.plot(ek_curve3[2].cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        # plt.legend()
        # plt.savefig(outf + "fault3.png")

        # plt.show()

        # plt.figure()
        # plt.title("Abnormal Mode III Filter")
        # plt.plot(util.low_pass_filter(ek_curve3[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
        # plt.plot(util.low_pass_filter(ek_curve3[1], curout_frequency, sampling_rate).cpu(), label='TE')
        # plt.plot(util.low_pass_filter(ek_curve3[2], curout_frequency, sampling_rate).cpu(), label='PI')
        # plt.ylim(-0.9, 0.9)
        # plt.legend()
        # plt.savefig(outf + "fault3_filter.png")

        # plt.show()

        # max_fault3 = torch.max(torch.cat([util.low_pass_filter(ek_curve3[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
        #                                 util.low_pass_filter(ek_curve3[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
        #                                 util.low_pass_filter(ek_curve3[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

        # # print("early detect of fault3:", max_fault3.gt(val_em0.item()).nonzero(as_tuple=True)[0][0])

        # max_x = torch.argmax(max_fault3)
        # max_y = torch.max(max_fault3[max_x])

        # plt.figure()
        # plt.title("Abnormal Mode III Error Curve")
        # plt.plot(max_fault3)
        # plt.ylim(-0.9, 0.9)
        # plt.scatter(max_x, max_y, color='red', s=50)
        # plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x + 2000, max_y - 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
        # plt.savefig(outf + "fault3_error_curve.png")
        # plt.show()

        plt.figure()
        plt.title("Comparison of Max Error Curves with Wcog,wmot12")
        plt.plot(max_normal, label='ek_wcog')
        plt.plot(max_fault1, label='ek_wmot1')
        plt.plot(max_fault2, label='ek_wmot2')
        # plt.plot(max_fault3, label='Abnormal Mode III')
        # plt.ylim(-0.9, 0.9)
        plt.legend()
        plt.savefig(outf + "comp_error_curve.png")

def data_process(df:pd.DataFrame, k):
    '''调用数据处理函数,完成滤波与归一化'''
    xk_realtime = DcD.urtracker_data_process(df)

    # 实时眼动数据
    xk_gaze = torch.cat(
        [
            xk_realtime["entire_raw_gaze_x"].unsqueeze(0),
            xk_realtime["entire_raw_gaze_y"].unsqueeze(0),
            # xk_realtime["PI_gaze"].unsqueeze(0),
        ],
        dim=0,
    )
    # print("xk_gaze shape:", xk_gaze.shape)
    # 实时运动数据
    xk_motion = torch.cat(
        [
            xk_realtime["entire_raw_x"].unsqueeze(0),
            xk_realtime["entire_raw_y"].unsqueeze(0),
            # xk_realtime["force_norm"].unsqueeze(0),
        ]
    )
    # print("xk_motion shape:", xk_motion.shape)

    # center_point = torch.tensor([540, 540])
    
    # xk_gaze_raw = torch.cat(((torch.tensor(df['Gaze_x'].values)-center_point[0]).unsqueeze(0),
    #                          (torch.tensor(df['Gaze_y'].values)-center_point[1]).unsqueeze(0)), dim=0)
    # xk_cursor_raw = torch.cat(((torch.tensor(df['cursor_x'].values)-center_point[0]).unsqueeze(0),
    #                            (torch.tensor(df['cursor_y'].values)-center_point[1]).unsqueeze(0)), dim=0)

    # xk_gaze[2,:] = xk_gaze[2,:]-4.5
    data_length = xk_gaze.shape[1]
    # data_raw_length = xk_cursor_raw.shape[1]
    # gaze_factors = torch.tensor([[0.004, 0.004, 0.002]], device='cuda').repeat(data_length, 1).T
    # motion_factors = torch.tensor([[0.004, 0.004, 0.002]], device='cuda').repeat(data_length, 1).T
    gaze_factors = torch.tensor([[0.004, 0.004]], device='cuda').repeat(data_length, 1).T
    motion_factors = torch.tensor([[0.004, 0.004]], device='cuda').repeat(data_length, 1).T
    xk_gaze = k * F.tanh(gaze_factors * xk_gaze)
    xk_motion = k * F.tanh(motion_factors * xk_motion)

    
    # torch.save({
    #     "xk_gaze" : xk_gaze,"xk_motion" : xk_motion, 'xk_cursor_raw' : xk_cursor_raw, 'xk_gaze_raw' : xk_gaze_raw,
    #             },
    #             "output/realtime_xk_filtered_{}.pt".format(time.time()))
    # print('realtime xk pt saved')

    return xk_gaze, xk_motion

def realtime_attention_detect_cc(df:pd.DataFrame, center, W_cog, W_mot1, W_mot2, draw_debug=False):
    '''对注意力运动状态四分类, 返回1: NORM 返回2: ABN-M 返回3"ABN-A 返回4: ABN-MA'''
    # 初始化网络参数
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    this_time = "Detect" + time.strftime("%m%d_%H%M")

    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda:0")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")

    k = 1.0
    eta = 0.2
    b = 0.2
    state_flag = 0
    # 导入数据
    print("Start Loading Data")
    # 预处理
    time_start_preprocess = time.time()
    xk_gaze,xk_motion = data_process(df, k)
    print('load data time:', time.time() - time_start_preprocess, 's')

    # 绘图debug
    # draw_debug = True
    if draw_debug:
        # plt.close()
        show_length = 400
        plt.figure("RT reatime")
        plt.plot(xk_gaze[0, :show_length].cpu(), label="gaze_reatime")
        plt.plot(xk_motion[0, :show_length].cpu(), label="motion_reatime")
        plt.legend()
        plt.show()

        plt.figure("TE realtime")
        plt.plot(xk_gaze[1, :show_length].cpu(), label="gaze")
        plt.plot(xk_motion[1, :show_length].cpu(), label="motion")
        plt.legend()
        plt.show()

        plt.figure("....realtime")
        plt.plot(xk_gaze[2, :show_length].cpu(), label="gaze")
        plt.plot(xk_motion[2, :show_length].cpu(), label="motion")
        plt.legend()
        plt.show()

    print("Data Loaded")

    # Start Detecting
    print("Start Detecting")
    
    # 最大匹配阈值
    tau_cog = 0.01
    tau_mot1 = 0.008
    tau_mot2 = 0.008
    '''计算e_cog'''
    time_start_detect = time.time()

    # # 比较优化后的计算
    e_cog_c = util.calc_max_error_cc(W_cog, xk_gaze, center, b, eta)
    tc1=time.time()
    print('cog detect time after:', tc1-time_start_detect)
    # print('Compare resluts before vs after : ', e_cog,'vs', e_cog_c)

    if e_cog_c <= tau_cog:
        e_mot1_c = util.calc_max_error_cc(W_mot1, xk_motion, center, b, eta)
        if e_mot1_c <= tau_mot1:
            state_flag = 1
            # return state_flag, e_cog_c, e_mot1_c # NORM
            return (
                int(state_flag),  # 转换为Python int
                float(e_cog_c), # 转换为Python float
                float(e_mot1_c)
            )
        else:
            state_flag = 2
            # return state_flag, e_cog_c, e_mot1_c # ABN-M
            return (
                int(state_flag),  # 转换为Python int
                float(e_cog_c), # 转换为Python float
                float(e_mot1_c)
            )
    else:
        e_mot2_c = util.calc_max_error_cc(W_mot2, xk_motion, center, b, eta)
        if e_mot2_c <= tau_mot2:
            state_flag = 3
            # return state_flag, e_cog_c, e_mot2_c # ABN-A
            return (
                int(state_flag),  # 转换为Python int
                float(e_cog_c), # 转换为Python float
                float(e_mot2_c)
            )
        else:
            state_flag = 4
            # return state_flag, e_cog_c, e_mot2_c # ABN-MA
            return (
                int(state_flag),  # 转换为Python int
                float(e_cog_c), # 转换为Python float
                float(e_mot2_c)
            )
    # val_em2, ek_curve_mot2, x_hat2, x_k2 = util.calc_max_error(W_mot2, xk_motion, center, b, eta)
    

    # # e_mot1 = torch.mean(ek_curve_mot1[:,10:-10] ** 2)
    # e_mot2 = torch.mean(ek_curve_mot2[:,10:-10] ** 2)
    # print('Detect time:', time.time()-time_start_detect)
    # curout_frequency = 0.1
    # sampling_rate = 20
    # # Drawfig = True
    # if draw_debug:

    #     outf = "experiment/urtracker/{}/".format(this_time)
    #     if not os.path.exists(outf): os.makedirs(outf, exist_ok=True)
    #     '''w_cog模型结果'''
    #     # 绘制ek_curve_cog
    #     plt.figure()
    #     plt.title("ek_curve_cog")
    #     plt.plot(ek_curve_cog[0].cpu(), label='ek_RT_gaze')
    #     plt.plot(ek_curve_cog[1].cpu(), label='ek_TE_gaze')
    #     plt.plot(ek_curve_cog[2].cpu(), label='ek_PI_gaze')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_curve_cog.png")
    #     plt.show()
    #     # 绘制ek_curvce三维最大值
    #     max_normal = torch.max(torch.cat([util.low_pass_filter(ek_curve_cog[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_cog[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_cog[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    #     max_x = torch.argmax(max_normal)
    #     max_y = torch.max(max_normal[max_x])
    #     plt.figure()
    #     plt.title("Max Error Curve cog")
    #     plt.plot(max_normal)
    #     # plt.ylim(-0.9, 0.9)
    #     plt.scatter(max_x, max_y, color='red', s=50)
    #     plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    #     plt.savefig(outf + "max_error_curve_cog.png")
    #     plt.show()

    #     plt.figure()
    #     plt.title("ek_curve_cog Filter")
    #     plt.plot(util.low_pass_filter(ek_curve_cog[0], curout_frequency, sampling_rate).cpu(), label='RT_gaze')
    #     plt.plot(util.low_pass_filter(ek_curve_cog[1], curout_frequency, sampling_rate).cpu(), label='TE_gaze')
    #     plt.plot(util.low_pass_filter(ek_curve_cog[2], curout_frequency, sampling_rate).cpu(), label='PI_gaze')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_cog_filter.png")
    #     plt.show()
    #     ''' w_mot1 输出结果'''
    #     # 绘制ek_curve_mot1
    #     plt.figure()
    #     plt.title("ek_curve_mot1")
    #     plt.plot(ek_curve_mot1[0].cpu(), label='RT')
    #     plt.plot(ek_curve_mot1[1].cpu(), label='TE')
    #     plt.plot(ek_curve_mot1[2].cpu(), label='force')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_curve_mot1.png")
    #     plt.show()

    #     plt.figure()
    #     plt.title("ek_curve_mot1_filter")
    #     plt.plot(util.low_pass_filter(ek_curve_mot1[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    #     plt.plot(util.low_pass_filter(ek_curve_mot1[1], curout_frequency, sampling_rate).cpu(), label='TE')
    #     plt.plot(util.low_pass_filter(ek_curve_mot1[2], curout_frequency, sampling_rate).cpu(), label='PI')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_curve_mot1_filter.png")
    #     plt.show()
    #     # 绘制最大值
    #     max_fault1 = torch.max(torch.cat([util.low_pass_filter(ek_curve_mot1[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_mot1[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_mot1[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    #     max_x = torch.argmax(max_fault1)
    #     max_y = torch.max(max_fault1[max_x])
    #     # print("early detect of fault1:", max_fault1.gt(0).nonzero(as_tuple=True)[0][0])
    #     plt.figure()
    #     plt.title("Abnormal Mode I Error Curve")
    #     plt.plot(max_fault1)
    #     # plt.ylim(-0.9, 0.9)
    #     plt.scatter(max_x, max_y, color='red', s=50)
    #     plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x + 2000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    #     print(max_fault1.gt(0.0114).nonzero(as_tuple=True))
    #     plt.savefig(outf + "fault1_error_curve.png")
    #     plt.show()
    #     '''mot2'''
    #     plt.figure()
    #     plt.title("ek_curve_mot2")
    #     plt.plot(ek_curve_mot2[0].cpu(), label='RT_r')
    #     plt.plot(ek_curve_mot2[1].cpu(), label='TE')
    #     plt.plot(ek_curve_mot2[2].cpu(), label='PI')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_curve_mot2.png")

    #     plt.show()

    #     plt.figure()
    #     plt.title("ek_curve_mot2_filter")
    #     plt.plot(util.low_pass_filter(ek_curve_mot2[0], curout_frequency, sampling_rate).cpu(), label='RT_r')
    #     plt.plot(util.low_pass_filter(ek_curve_mot2[1], curout_frequency, sampling_rate).cpu(), label='TE')
    #     plt.plot(util.low_pass_filter(ek_curve_mot2[2], curout_frequency, sampling_rate).cpu(), label='PI')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "ek_curve_mot2_filter.png")
    #     plt.show()

    #     max_fault2 = torch.max(torch.cat([util.low_pass_filter(ek_curve_mot2[0], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_mot2[1], curout_frequency, sampling_rate).cpu().unsqueeze(0),
    #                                     util.low_pass_filter(ek_curve_mot2[2], curout_frequency, sampling_rate).cpu().unsqueeze(0)], dim=0), dim=0).values

    #     # print("early detect of fault2:", max_fault2.gt(val_em0.item()).nonzero(as_tuple=True)[0][0])

    #     max_x = torch.argmax(max_fault2)
    #     max_y = torch.max(max_fault2[max_x])

    #     plt.figure()
    #     plt.title("Abnormal Mode II Error Curve")
    #     plt.plot(max_fault2)
    #     # plt.ylim(-0.9, 0.9)
    #     plt.scatter(max_x, max_y, color='red', s=50)
    #     plt.annotate(f'max: ({max_x},{max_y:.4f})', xy=(max_x, max_y), xytext=(max_x - 4000, max_y + 0.1), arrowprops=dict(facecolor='red', shrink=0.05))
    #     plt.savefig(outf + "fault2_error_curve.png")
    #     plt.show()

    #     plt.figure()
    #     plt.title("Comparison of Max Error Curves with Wcog,wmot12")
    #     plt.plot(max_normal, label='ek_wcog')
    #     plt.plot(max_fault1, label='ek_wmot1')
    #     plt.plot(max_fault2, label='ek_wmot2')
    #     # plt.plot(max_fault3, label='Abnormal Mode III')
    #     # plt.ylim(-0.9, 0.9)
    #     plt.legend()
    #     plt.savefig(outf + "comp_error_curve.png")
    
    # state_flag = 0
    # if e_cog <= tau_cog:
    #     if e_mot1 <= tau_mot1:
    #         state_flag = 1
    #         return state_flag, e_cog, e_mot1 # NORM
    #     else:
    #         state_flag = 2
    #         return state_flag, e_cog, e_mot1 # ABN-M
    # else:
    #     if e_mot2 <= tau_mot2:
    #         state_flag = 3
    #         return state_flag, e_cog, e_mot2 # ABN-A
    #     else:
    #         state_flag = 4
    #         return state_flag, e_cog, e_mot2 # ABN-MA
         

if __name__ == '__main__':
    realtime_attention_detect(r'output\cc_NORM_02.xlsx', draw_debug=True)
    realtime_attention_detect(r'output\cc_ABN_A_02.xlsx', draw_debug=True)
    realtime_attention_detect(r'output\cc_ABN_M_02.xlsx', draw_debug=True)
    realtime_attention_detect(r'output\cc_ABN_MA_02.xlsx', draw_debug=True)
    '''测试realtime'''

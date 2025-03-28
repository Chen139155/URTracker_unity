import torch
import data_generator_torch as dg
import DL_DataProcess as DD
from RBFNN_lib_torch import RBFNN
from DL_lib_torch import DynamicLearning
import matplotlib.pyplot as plt
import torch.nn.functional as F
import util
# import wandb
import time
import argparse
import os

this_time = time.time()
time_tag = time.strftime("%m%d_%H%M")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda:0")
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=1.0)
parser.add_argument('--eta', default=0.1)
parser.add_argument('--b', default=0.2)
parser.add_argument('--k', default=1.0) # RBFNN边界范围
parser.add_argument('--epoch', default=100)
parser.add_argument('--Ao', default=0.1)
parser.add_argument('--output_dir', type=str, default='urtracker')
parser.add_argument('--avg_length', default=5)

opt = parser.parse_args()

k = opt.k
eta = opt.eta
gamma = opt.gamma
b = opt.b
epoch = opt.epoch
Ao = opt.Ao * torch.eye(3)

# 生成网格
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
    xk_normal_attention_motion = DD.get_urtracker_data(
        path=r"./output/cc_NORM_01.xlsx",type = "Norm_AM"
    )
    xk_abnormal_attention_motion = DD.get_urtracker_data(
        path=r"./output/cc_ABN_A_01.xlsx", type ="ABN_A"
    )
    
    # 注意力正常的注视数据
    xk_normal_gaze = torch.cat(
        [
            xk_normal_attention_motion["entire_raw_gaze_x"].unsqueeze(0),
            xk_normal_attention_motion["entire_raw_gaze_y"].unsqueeze(0),
            xk_normal_attention_motion["PI_gaze"].unsqueeze(0),
        ],
        dim=0,
    )
    print("xk_normal_gaze shape:", xk_normal_gaze.shape) # torch.Size([3, 4500])
    # 注意力正常的运动数据
    xk_normal_motion = torch.cat(
        [
            xk_normal_attention_motion["entire_raw_x"].unsqueeze(0),
            xk_normal_attention_motion["entire_raw_y"].unsqueeze(0),
            xk_normal_attention_motion["force_norm"].unsqueeze(0),
        ]
    )
    print("xk_normal_motion shape:", xk_normal_motion.shape)
    # 注意力异常的正常运动数据
    xk_abnormal_motion = torch.cat(
        [
            xk_abnormal_attention_motion["entire_raw_x"].unsqueeze(0),
            xk_abnormal_attention_motion["entire_raw_y"].unsqueeze(0),
            xk_abnormal_attention_motion["force_norm"].unsqueeze(0),
        ]
    )
    print("xk_abnormal_motion shape:", xk_abnormal_motion.shape)


    



    torch.save(xk_normal_gaze, "output/xk_normal_gaze_{}.pt".format(type))
    torch.save(xk_normal_motion, "output/xk_normal_motion_{}.pt".format(type))
    torch.save(xk_abnormal_motion, "output/xk_abnormal_{}.pt".format(type))
    # torch.save(xk_m1_fault1, "output/xk_fault3_robio_{}.pt".format(type))
else:
    xk_normal_gaze = torch.load("output/xk_normal_gaze_{}.pt".format(type))
    xk_normal_motion = torch.load("output/xk_normal_motion_{}.pt".format(type))
    xk_abnormal_motion = torch.load("output/xk_abnormal_{}.pt".format(type))
    # xk_m1_fault1 = torch.load("output/xk_fault3_robio_{}.pt".format(type))

xk_normal_gaze[2,:] = xk_normal_gaze[2,:]-4.5
# 裁切去掉每行最前面的 500 个元素和最后面的 500 个元素
xk_normal_gaze = xk_normal_gaze[:, 465:3950-465] 
xk_normal_motion = xk_normal_motion[:, 465:3950-465]
xk_abnormal_motion = xk_abnormal_motion[:, 465:3950-465]

data_length = xk_normal_gaze.shape[1]
gaze_factors = torch.tensor([[0.004, 0.004, 0.4]], device='cuda').repeat(data_length, 1).T
motion_factors = torch.tensor([[0.004, 0.004, 0.02]], device='cuda').repeat(data_length, 1).T
# vec_factor = 0.1

# # 裁切去掉每行最前面的 500 个元素和最后面的 500 个元素
# xk_normal_gaze = xk_normal_gaze[:, 465:3950-465] 
# xk_normal_motion = xk_normal_motion[:, 465:3950-465]
# xk_abnormal_motion = xk_abnormal_motion[:, 465:3950-465]
# 将数据缩放到（-1，1）范围
xk_normal_gaze = k * F.tanh(gaze_factors * xk_normal_gaze)
xk_normal_motion = k * F.tanh(motion_factors * xk_normal_motion)
xk_abnormal_motion = k * F.tanh(motion_factors * xk_abnormal_motion)
# xk_m1_fault1 = k * F.tanh(factors * xk_m1_fault1)
draw_debug = False

if draw_debug:
    # plt.close()
    show_length = 1000
    plt.figure("gaze curve")
    plt.plot(xk_normal_gaze[0, :].cpu(), label="gazex")
    plt.plot(xk_normal_gaze[1, :].cpu(), label="gazey")
    plt.plot(xk_normal_gaze[2, :].cpu(), label="PI")
    plt.legend()
    plt.show()

    plt.figure("motion1 curve")
    plt.plot(xk_normal_motion[0, :].cpu(), label="x")
    plt.plot(xk_normal_motion[1, :].cpu(), label="y")
    plt.plot(xk_normal_motion[2, :].cpu(), label="force")
    plt.legend()
    plt.show()

    plt.figure("motion2 curve")
    plt.plot(xk_abnormal_motion[0, :].cpu(), label="x")
    plt.plot(xk_abnormal_motion[1, :].cpu(), label="y")
    plt.plot(xk_abnormal_motion[2, :].cpu(), label="force")
    plt.legend()
    plt.show()

# xk_m1_fault1 = xk_m1_fault1[:, 500:-500]

x0_normal = xk_normal_gaze[:, 0].reshape(-1, 1)  # 列向量
x0_fault1 = xk_normal_motion[:, 0].reshape(-1, 1)
x0_fault2 = xk_abnormal_motion[:, 0].reshape(-1, 1)
# x0_fault3 = xk_m1_fault1[:, 0].reshape(-1, 1)

n = xk_normal_gaze.size(0)
steps = xk_normal_gaze.size(1)
print("Data Loaded")



print("Start Learning")

# Wb0 = torch.load("experiment/tobii_ori/1714293909.0843563/trained_models/iden_trackball_lowpass_final.pt")["Wb0"].cuda()

resume_model = ""

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_normal,
                                                         xk_normal_gaze,
                                                         N,
                                                         n,
                                                         steps,
                                                         center,
                                                         b,
                                                         eta,
                                                         Ao,
                                                         epoch,
                                                         opt.avg_length,
                                                         outf=opt.output_dir,
                                                         mode="NOM_A",
                                                         time_tag=time_tag,
                                                         resume=resume_model)

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_fault1,
                                                         xk_normal_motion,
                                                         N,
                                                         n,
                                                         steps,
                                                         center,
                                                         b,
                                                         eta,
                                                         Ao,
                                                         epoch,
                                                         opt.avg_length,
                                                         outf=opt.output_dir,
                                                         mode="NOM_M",
                                                         time_tag=time_tag,
                                                         resume=resume_model)

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_fault2,
                                                         xk_abnormal_motion,
                                                         N,
                                                         n,
                                                         steps,
                                                         center,
                                                         b,
                                                         eta,
                                                         Ao,
                                                         epoch,
                                                         opt.avg_length,
                                                         outf=opt.output_dir,
                                                         mode="ABN_M",
                                                         time_tag=time_tag,
                                                         resume=resume_model)
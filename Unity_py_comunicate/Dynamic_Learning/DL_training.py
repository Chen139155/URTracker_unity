import torch
import data_generator_torch as dg
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
parser.add_argument('--k', default=1.0)
parser.add_argument('--epoch', default=40)
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

print("Start Loading Data")
type = "sin"
reprocess_data = False
if reprocess_data:
    xk_mode2_normal = dg.get_trackball_data_robio(
        path=r"output/1727011004.2450852_mode2_normal.xlsx",type = "_mode2_normal"
    )
    xk_mode2_fault1 = dg.get_trackball_data_robio(
        path=r"output/1727011241.330176_mode2_fault1.xlsx", type ="_mode2_fault1"
    )
    xk_mode1_normal = dg.get_trackball_data_robio(
        path=r"output/1727011893.1889017_mode1_normal.xlsx", type="_mode1_normal"
    )
    xk_mode1_fault1 = dg.get_trackball_data_robio(
        path=r"output/1727011525.9612145_mode1_fault1.xlsx", type="_mode1_fault1"
    )

    xk_m2_normal = torch.cat(
        [
            xk_mode2_normal["RT_r"].unsqueeze(0),
            xk_mode2_normal["TE"].unsqueeze(0),
            xk_mode2_normal["PI"].unsqueeze(0),
        ],
        dim=0,
    )
    print("xk_normal shape:", xk_m2_normal.shape) # torch.Size([3, 4500])

    xk_m2_fault1 = torch.cat(
        [
            xk_mode2_fault1["RT_r"].unsqueeze(0),
            xk_mode2_fault1["TE"].unsqueeze(0),
            xk_mode2_fault1["PI"].unsqueeze(0),
        ]
    )

    print("xk_fault shape:", xk_m2_fault1.shape)

    xk_m1_normal = torch.cat(
        [
            xk_mode1_normal["RT_r"].unsqueeze(0),
            xk_mode1_normal["TE"].unsqueeze(0),
            xk_mode1_normal["PI"].unsqueeze(0),
        ]
    )

    print("xk_fault2 shape:", xk_m1_normal.shape)

    xk_m1_fault1 = torch.cat(
        [
            xk_mode1_fault1["RT_r"].unsqueeze(0),
            xk_mode1_fault1["TE"].unsqueeze(0),
            xk_mode1_fault1["PI"].unsqueeze(0),
        ]
    )

    print("xk_fault3 shape:", xk_m1_fault1.shape)

    draw_debug = False

    if draw_debug:
        # plt.close()
        show_length = 1000
        plt.figure("x curve")
        plt.plot(xk_m2_normal[0, :show_length].cpu(), label="normal")
        plt.plot(xk_m2_fault1[0, :show_length].cpu(), label="fault1")
        plt.plot(xk_m1_normal[0, :show_length].cpu(), label="fault2")
        plt.plot(xk_m1_fault1[0, :show_length].cpu(), label="fault3")
        plt.legend()
        plt.show()

        plt.figure("y curve")
        plt.plot(xk_m2_normal[1, :show_length].cpu(), label="normal")
        plt.plot(xk_m2_fault1[1, :show_length].cpu(), label="fault1")
        plt.plot(xk_m1_normal[1, :show_length].cpu(), label="fault2")
        plt.plot(xk_m1_fault1[1, :show_length].cpu(), label="fault3")
        plt.legend()
        plt.show()

        plt.figure("vx curve")
        plt.plot(xk_m2_normal[2, :show_length].cpu(), label="normal")
        plt.plot(xk_m2_fault1[2, :show_length].cpu(), label="fault1")
        plt.plot(xk_m1_normal[2, :show_length].cpu(), label="fault2")
        plt.plot(xk_m1_fault1[2, :show_length].cpu(), label="fault3")
        plt.legend()
        plt.show()

        plt.figure("vy curve")
        plt.plot(xk_m2_normal[3, :show_length].cpu(), label="normal")
        plt.plot(xk_m2_fault1[3, :show_length].cpu(), label="fault1")
        plt.plot(xk_m1_normal[3, :show_length].cpu(), label="fault2")
        plt.plot(xk_m1_fault1[3, :show_length].cpu(), label="fault3")
        plt.legend()
        plt.show()

    torch.save(xk_m2_normal, "output/xk_normal_robio_{}.pt".format(type))
    torch.save(xk_m2_fault1, "output/xk_fault1_robio_{}.pt".format(type))
    torch.save(xk_m1_normal, "output/xk_fault2_robio_{}.pt".format(type))
    torch.save(xk_m1_fault1, "output/xk_fault3_robio_{}.pt".format(type))
else:
    xk_m2_normal = torch.load("output/xk_normal_robio_{}.pt".format(type))
    xk_m2_fault1 = torch.load("output/xk_fault1_robio_{}.pt".format(type))
    xk_m1_normal = torch.load("output/xk_fault2_robio_{}.pt".format(type))
    xk_m1_fault1 = torch.load("output/xk_fault3_robio_{}.pt".format(type))

factors = torch.tensor([[0.03, 0.04, 0.4]], device='cuda').repeat(4500, 1).T
vec_factor = 0.1

xk_m2_normal = k * F.tanh(factors * xk_m2_normal)
xk_m2_fault1 = k * F.tanh(factors * xk_m2_fault1)
xk_m1_normal = k * F.tanh(factors * xk_m1_normal)
xk_m1_fault1 = k * F.tanh(factors * xk_m1_fault1)

xk_m2_normal = xk_m2_normal[:, 500:-500]
xk_m2_fault1 = xk_m2_fault1[:, 500:-500]
xk_m1_normal = xk_m1_normal[:, 500:-500]
xk_m1_fault1 = xk_m1_fault1[:, 500:-500]

x0_normal = xk_m2_normal[:, 0].reshape(-1, 1)  # 列向量
x0_fault1 = xk_m2_fault1[:, 0].reshape(-1, 1)
x0_fault2 = xk_m1_normal[:, 0].reshape(-1, 1)
x0_fault3 = xk_m1_fault1[:, 0].reshape(-1, 1)

n = xk_m2_normal.size(0)
steps = xk_m2_normal.size(1)
print("Data Loaded")

print("Init Wandb")
# wandb.init(
#     project="DynamicLearning",
#     entity="mufengjun260",
#     name="trackball_fault_identify_lowpass_tobii",
#     config={
#         "Ao": Ao,
#         "gamma": gamma,
#         "eta": eta,
#         "b": b,
#         "k": k,
#         "steps": steps,
#         "N": N,
#         "n": n,
#         "pos_vector": pos_factor,
#         "vec_vector": vec_factor,
#         "epoch": epoch,
#         "reprocess_data": reprocess_data,
#     },
# )

print("Wandb Initialed")

print("Start Learning")

# Wb0 = torch.load("experiment/tobii_ori/1714293909.0843563/trained_models/iden_trackball_lowpass_final.pt")["Wb0"].cuda()

resume_model = ""

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_normal,
                                                         xk_m2_normal,
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
                                                         mode="m2_normal",
                                                         time_tag=this_time,
                                                         resume=resume_model)

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_fault1,
                                                         xk_m2_fault1,
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
                                                         mode="m2_fault1",
                                                         time_tag=this_time,
                                                         resume=resume_model)

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_fault2,
                                                         xk_m1_normal,
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
                                                         mode="m1_normal",
                                                         time_tag=this_time,
                                                         resume=resume_model)

Wb0, Wb0_list, avg_Wb0 = util.start_dynamic_learning_old(x0_fault3,
                                                         xk_m1_fault1,
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
                                                         mode="m1_fault1",
                                                         time_tag=this_time,
                                                         resume=resume_model)

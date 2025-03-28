import torch
from RBFNN_lib_torch import RBFNN


class DynamicLearning:
    @staticmethod
    def learn(data_0, data_input, W0, steps, center, eta, Ao, gamma=0.04):
        # 初始化参数
        xh = data_0.clone()
        x = data_0.clone()
        S = RBFNN.inf(x, center, eta)
        W = W0.clone()
        n = xh.size(0)
        N = center.size(1)

        # 初始化记录神经网络状态的数组
        sk_list = torch.zeros(N, steps)
        xh_list = torch.zeros(n, steps)
        ek_list = torch.zeros(n, steps)
        Wk_list = torch.zeros(N, steps, n)

        # 循环遍历每个时间步
        for k in range(steps):
            sk_list[:, k] = S.view(-1)
            xh1 = Ao.mm(xh - x.reshape(-1, 1)) + W.t().mm(S)  # (5~7) -> (9)
            # 更新x为下一步的输入数据
            x = data_input[:, k]
            W = W - gamma * torch.outer(S.squeeze(), (xh1.squeeze() - x))  # (8)

            xh = xh1
            xh_list[:, k] = xh.squeeze()
            ek_list[:, k] = (xh.squeeze() - x).view(-1)
            Wk_list[:, k, :] = W
            S = RBFNN.inf(x, center, eta)

        # 学习神经网络模型
        wb_list = Wk_list[:, -100:, :].mean(dim=1)  # (11)

        return Wk_list, xh_list, ek_list, wb_list, sk_list

    @staticmethod
    def learn_mini(data_0, data_input, W0, steps, center, eta, Ao, gamma=0.04):
        # 初始化参数
        xh = data_0.clone()
        x = data_0.clone()
        S = RBFNN.inf(x, center, eta)
        W = W0.clone()
        n = xh.size(0)
        N = center.size(1)

        # 初始化记录神经网络状态的数组
        sk_list = torch.zeros(N, steps)
        xh_list = torch.zeros(n, steps)
        ek_list = torch.zeros(n, steps)
        # Wk_list = torch.zeros(N, steps, n)
        wb_list = torch.zeros(N, n)
        # 循环遍历每个时间步
        for k in range(1, steps):
            # sk_list[:, k] = S.view(-1)
            xh1 = Ao.mm(xh - x.reshape(-1, 1)) + W.t().mm(S)  # (5~7) -> (9)
            # 更新x为下一步的输入数据
            x = data_input[:, k]
            W = W - gamma * torch.outer(S.squeeze(), (xh1.squeeze() - x))  # (8)
            xh = xh1
            xh_list[:, k] = xh.squeeze()
            ek_list[:, k] = (xh.squeeze() - x).view(-1)
            # online update wb
            wb_list += W / (steps - 1)
            # Wk_list[:, k, :] = W
            S = RBFNN.inf(x, center, eta)

        # 学习神经网络模型
        # wb_list = Wk_list[:, :, :].mean(dim=1) #  (11)

        return xh_list, ek_list, wb_list

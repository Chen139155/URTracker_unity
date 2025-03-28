import torch

class RBFNN():
    @staticmethod
    def inf(x, center, eta):
        r = x.reshape(-1, 1).expand_as(center) - center  # 使用 PyTorch 的广播扩展 x
        ST = torch.exp(-torch.sum(r ** 2, axis=0) / (eta ** 2))  # 计算径向基函数
        S = ST.reshape(-1, 1)  # 调整ST的形状，使其成为列向量

        wanna_draw=False
        if wanna_draw:
            import matplotlib.pyplot as plt
            import numpy as np
            x=np.linspace(0,5654,5655)
            y=S.numpy()                                                                                     
            plt.plot(x,y)
            plt.show()
        return S

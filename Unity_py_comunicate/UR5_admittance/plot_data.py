import matplotlib.pyplot as plt
import pandas
import openpyxl
import numpy as np

if __name__=='__main__':
    pandas.set_option('display.notebook_repr_html',False)
    df = pandas.read_excel(io=r'F:\Users\chenchen\Documents\课程\UR5\UR5_admittance\output\1715580972.8497684.xlsx')
    df['time'] -= df['time'][0]
    # print(df['time'][0])
    # print(df)
    # t=df['time']
    # pose_x=df['pose_x']
    # pose_y=df['pose_y']
    # pose_z=df['pose_z']
    
    fig=plt.figure(1,figsize=(9, 6))
    ax1=fig.add_subplot(311) # 创建图实例
    ax1.plot('time','pose_x',data=df,label='pose_x')
    ax1.plot('time','pose_y',data=df,label='pose_y')
    ax2=fig.add_subplot(312)
    ax2.plot('time','linear_x',data=df,label='linear_x')
    ax2.plot('time','linear_y',data=df,label='linear_y')
    ax3=fig.add_subplot(313)
    ax3.plot('time','Hex_x',data=df,label='Hex_x')
    ax3.plot('time','Hex_y',data=df,label='Hex_y')
    # y3 = x ** 3
    # ax.plot(x, y3, label='cubic') # 作y3 = x^3 图，并标记此线名为cubic
    plt.legend()
    plt.show()


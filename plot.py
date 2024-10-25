import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径

class kAssessor():
    def assess_k(self,k_real_values,k_est_values):
        '''LMSELoss'''
        total_loss = 0.0
        for k_est, k_real in zip(k_est_values[1:], k_real_values[1:]):
            x = k_est / k_real
            loss = math.log(k_est/k_real)**2
            total_loss += loss
        lmseloss = total_loss / len(k_real_values[1:])
        '''RDSELoss'''
        total_loss = 0.0
        for k_est, k_real in zip(k_est_values[1:], k_real_values[1:]):
            x = k_est / k_real
            loss = x**2 + 1/(x**2) - 2
            total_loss += loss
        rdseloss = total_loss / len(k_real_values[1:])
        return lmseloss,rdseloss

def evaluate(time_values,F_real_values,F_goal_values,k_real_values,k_est_values,pos_values,plot=True):
    # 计算指标
    # 对于图1，可以考虑计算均方误差（MSE）或其他跟踪误差指标
    mse = sum((F_real_values[i] - F_goal_values[i])**2 for i in range(len(F_real_values))) / len(F_real_values)
    print(f'F_real 和 F_goal 的均方误差 (MSE): {mse}')

    # 对于图3，可以考虑计算刚度估计误差
    k_criterion=kAssessor()
    k_lmse,k_rdse = k_criterion.assess_k(k_real_values,k_est_values)
    print(f'k_real 和 k_est 的对数均方误差 (LMSE): {k_lmse}')
    print(f'k_real 和 k_est 的相对平方对勾误差 (RDSE): {k_rdse}')

    if plot:
        # 绘制四张图
        plt.figure(figsize=(8, 8))

        # 图1: F_real 和 F_goal vs 时间
        plt.subplot(2, 2, 1)
        plt.plot(time_values, F_real_values)#, label='F_N')
        # plt.plot(time_values, F_goal_values, label='F_goal')
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('F_N (N)')
        plt.title('Curve of Grip Force Over Time')

        # 图2: F_real 和 F_goal vs pos
        plt.subplot(2, 2, 2)
        plt.plot(pos_values, F_real_values)#, label='F_real')
        # plt.plot(pos_values, F_goal_values, label='F_goal')
        plt.legend()
        plt.xlabel('x (mm)')
        plt.ylabel('F_N (N)')
        plt.title('Simulated Force-Deformation Curve')

        # 图3: k_real 和 k_est vs 时间
        plt.subplot(2, 2, 3)
        plt.plot(time_values, k_real_values, label='k_real')
        plt.plot(time_values, k_est_values, label='k_est')
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('k (100N/mm)')
        plt.title('k_real and k_est vs time')

        # 图4: k_est/k_real-1 vs 时间
        x=(np.array(k_est_values)/np.array(k_real_values))
        lse_values=(np.log(x)**2)
        rdse_values=x**2 + 1/(x**2) - 2
        plt.subplot(2, 2, 4)
        plt.plot(time_values, lse_values, label='LSELoss', color='blue')
        plt.xlabel('time (s)')
        plt.ylabel('LSELoss', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')
        plt.title('LSELoss vs time')

        # Create a second y-axis
        plt.twinx()
        plt.plot(time_values, rdse_values, label='RDSELoss', color='red')
        plt.ylabel('RDSELoss', color='red')
        plt.tick_params(axis='y', labelcolor='red')

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.show()
    return mse,k_lmse,k_rdse

if __name__=='__main__':
    # 读取数据
    data_file = curr_path+'/output.txt'
    with open(data_file, "r") as file:
        lines = file.readlines()

    # 解析数据
    time_values = []
    F_real_values = []
    F_goal_values = []
    k_real_values = []
    k_est_values = []
    pos_values = []

    for line in lines:
        # 假设数据之间使用空格分隔
        values = line.strip().split()

        # 将字符串转换为浮点数
        F_real = float(values[0])
        F_goal = float(values[1])
        k_real = float(values[2])
        k_est = float(values[3])
        pos = float(values[4])

        F_real_values.append(F_real)
        F_goal_values.append(F_goal)
        k_real_values.append(k_real)
        k_est_values.append(k_est)
        pos_values.append(pos)

        # 时间间隔为10毫秒
        time_values.append(len(time_values) * 0.01)
    
    evaluate(time_values,F_real_values,F_goal_values,k_real_values,k_est_values,pos_values,plot=True)




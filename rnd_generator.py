import numpy as np
from k_simulator import Elastomer0
from k_simulator import Elastomer1
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import math

def smooth_curve(x_values, y_values, kind='linear'):
    # 创建插值函数
    interp_func = interp1d(x_values, y_values, kind=kind, fill_value="extrapolate")

    return interp_func

def generate_random_function(t_range=(0, 20), num_segments=75, max_slope=5, function_value_range=(0.1, 5), flat_probability=0.1):
    # np.random.seed(3 30)
    gamma=np.random.uniform(0, 1)
    # 生成随机的时间点
    time_points = np.sort(np.random.uniform(t_range[0], t_range[1], num_segments - 1))

    # 生成随机的斜率，限制在[-max_slope, max_slope]范围内
    slopes = np.random.uniform(-max_slope, max_slope, num_segments)

    # 生成随机的函数值范围
    function_values = [0]
    for i in range(num_segments - 1):
        # 以一定概率生成平坦的折线段
        if np.random.rand() < flat_probability:
            slopes[i]=np.random.uniform(-0.3, 0.3)
        # 限制斜率在[-max_slope, max_slope]范围内
        slope = np.clip(slopes[i], -max_slope, max_slope)
        delta_t = time_points[i] - time_points[i - 1] if i > 0 else time_points[i]
        delta_y = slope * delta_t
        new_value = function_values[-1] + delta_y

        # 限制函数值范围在[0.1, 5]范围内
        new_value = np.clip(new_value, function_value_range[0], function_value_range[1])
        
        function_values.append(new_value)

    function_values.append(0)

    # 将时间点和函数值连接起来，形成折线函数
    time_values = np.concatenate(([t_range[0]], time_points, [t_range[1]]))
    function_values = np.array(function_values)

    # 去除重复的时间点
    unique_time_values, unique_indices = np.unique(time_values, return_index=True)
    function_values=function_values[unique_indices]

    F_values=[]
    F_values.append(np.interp(0,unique_time_values,function_values))
    for i in range(1,int(t_range[1]/delta_t)):
        F_values.append((1-gamma)*np.interp(delta_t*i,unique_time_values,function_values)+gamma*F_values[-1])        
    return np.linspace(0, t_range[1], int(t_range[1]/delta_t)),F_values

class Generater():
    def generate_func(self):
        x_data,y_data=generate_random_function()
        interpolation_func = smooth_curve(x_data, y_data)
        return interpolation_func
    def rnd_hand_config(self):
        '''帮我用python写一个随机数生成器，共需要生成t0，F0，k1，k2，x1，x2六个随机数，具体要求如下：
        1.这六个量的期望值为t0=1,F0=1,k1=0.1,k2=0.167,xa=1.5*100,xb=3*100
        2.这六个量需要被严格地限制范围，每个量与其期望值的差都不能超过其期望值的50%'''
        # 期望值
        mean_t0 = 1
        mean_F0 = 1
        mean_k1 = 0.1
        mean_k2 = 0.167
        mean_xa = 1.5 * 100
        mean_xb = 3 * 100

        # 生成随机数
        t0 = np.random.uniform(0.8 * mean_t0, 1.2 * mean_t0)  # 期望值的50%范围内均匀分布
        F0 = np.random.uniform(0.8 * mean_F0, 1.2 * mean_F0)
        k1 = np.random.uniform(0.8 * mean_k1, 1.2 * mean_k1)
        k2 = np.random.uniform(0.8 * mean_k2, 1.2 * mean_k2)
        xa = np.random.uniform(0.8 * mean_xa, 1.2 * mean_xa)
        xb = np.random.uniform(0.8 * mean_xb, 1.2 * mean_xb)

        return t0, F0, k1, k2, xa, xb
   
    def generate_elas(self):
        t0, F0, k1, k2, x1, x2 = self.rnd_hand_config()
        hand=Elastomer0(delta_t=0.01,t0=t0,F0=F0,k1=k1,k2=k2,xa=x1,xb=x2)
        balloon=Elastomer1(delta_t=0.01)  
        return hand,balloon     

if __name__=='__main__':
    g=Generater()
    interpolation_func = g.generate_func()

    # 给定横坐标值，得到纵坐标值
    x_new = np.linspace(0, 20, 100)
    y_new = interpolation_func(x_new)

    # 绘制原始数据点和光滑连接的曲线
    plt.figure(figsize=(4, 4))
    plt.plot(x_new, y_new, color='red')
    plt.legend()
    plt.xlabel('t(s)')
    plt.ylabel('F_N(N)')
    plt.title('Curve of Grip Force Over Time')
    plt.show()
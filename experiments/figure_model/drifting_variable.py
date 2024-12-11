import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Drifter:
    def __init__(self,start, upper_bound, lower_bound, max_slope, min_slope,delta_t=0.01,max_t=20):
        self.num =np.random.randint(2, 4*max_t)
        self.gamma=1-math.pow(10,np.random.randint(-5, 1))
        self.delta_t=delta_t
        self.max_t=max_t
        self.t=0
        self.start = start
        self.now_x=start
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.max_slope = max_slope
        self.min_slope = min_slope
        self.t_values = np.sort(np.concatenate(([0.0], np.random.uniform(0.0, max_t, size=self.num - 2),[ max_t])))
        self.x_values = np.empty(self.num)
        self.x_values[0] = start
        self._generate_points()
    def _generate_points(self):
        for i in range(1, self.num):
            slope = np.random.uniform(self.min_slope, self.max_slope)
            delta_t = self.t_values[i] - self.t_values[i - 1]
            delta_x = slope * delta_t
            next_point = self.x_values[i - 1] + delta_x

            # 确保点在上下界之内
            if next_point >= self.upper_bound:
                next_point = self.upper_bound
            elif next_point <= self.lower_bound:
                next_point = self.lower_bound

            self.x_values[i] = next_point
    def get_new(self):
        new_x=np.interp(self.t,self.t_values,self.x_values)
        self.t+=self.delta_t
        new_x=self.gamma*self.now_x+(1-self.gamma)*new_x
        self.now_x=new_x
        return new_x
    def get_all(self):
        x_real=[]
        for i in range(int(self.max_t/self.delta_t)):
            x_real.append(self.get_new())        
        return np.linspace(0, self.max_t, int(self.max_t/self.delta_t)),x_real
    def plot(self):
        # print(self.gamma)
        t,x_real=self.get_all()
        plt.plot(t,x_real, marker='o')
        plt.xlabel('Time')
        plt.ylabel('X Value')
        plt.title('Drift Data Points')
        plt.grid(True)
        plt.show()

if __name__=='__main__':
    # 创建Drift类的实例
    drift_instance = Drifter(start=5, upper_bound=6, lower_bound=4, max_slope=0.5, min_slope=-0.5,max_t=30)

    # 绘制数据点
    drift_instance.plot()

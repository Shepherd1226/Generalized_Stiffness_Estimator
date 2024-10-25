import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from drifting_variable import Drifter
import matplotlib.pyplot as plt

def smooth_curve(x_values, y_values, kind='linear'):
    # 创建插值函数
    interp_func = interp1d(x_values, y_values, kind=kind, fill_value="extrapolate")
    return interp_func

class F_Pos_simulator():
    def __init__(self,delta_t=0.01) -> None:
        self.delta_t=delta_t
    def generate(self,plot=False):
        # np.random.seed(113)
        # 步骤1：随机生成数据点
        self.num_points = np.random.randint(2, 10)  # 随机生成2到10个数据点

        # 生成剩余的随机数据点
        self.F_values = np.sort(np.concatenate(([0.0], np.random.uniform(0.0, 10.0, size=self.num_points - 2),[10.0])))
        cliffy_probability=0.2
        for i in range(self.num_points - 2):
            # 以一定概率生成陡峭的折线段
            if np.random.rand() < cliffy_probability:
                self.F_values[i+1]=np.random.uniform(0.01, 0.3)+self.F_values[i]

        # 生成对应的导数值
        self.log_der_values = (np.random.uniform(np.log(1/0.07), np.log(1000), size=self.num_points))

        self.F_drifter=np.empty(self.num_points,dtype=object)
        self.log_der_drifter=np.empty(self.num_points,dtype=object)
        for i in range(self.num_points):
            self.F_drifter[i]=Drifter(self.F_values[i],min(self.F_values[i]*1.1,10.0),max(self.F_values[i]*0.9,0.0),0.1,-0.1)
            self.log_der_drifter[i]=Drifter(self.log_der_values[i],self.log_der_values[i]*1.1,self.log_der_values[i]*0.9,self.log_der_values[i]*0.1*0.1,-self.log_der_values[i]*0.1*0.1)
        # 积分初始值
        self.x_start=0
        max_pos = np.trapz(np.interp(np.arange(0, 10, 0.01),self.F_values,np.exp(self.log_der_values)), dx=0.01) + self.x_start
        self.x_drifter=Drifter(self.x_start,max_pos*0.04,-max_pos*0.02,max_pos*0.04*0.1,-max_pos*0.02*0.1)

    def step(self):
        self.x_start=self.x_drifter.get_new()
        for i in range(self.num_points):
            self.F_values[i]=self.F_drifter[i].get_new()
            self.log_der_values[i]=self.log_der_drifter[i].get_new()

    def pos_F_func(self,F):
        # 步骤2：使用插值法连接数据点，得到光滑曲线
        der=np.interp(F,self.F_values,np.exp(self.log_der_values))

        # 根据初始条件f(x0=0)=0生成f(x)
        pos_F = np.trapz(np.interp(np.arange(0, F, 0.01),self.F_values,np.exp(self.log_der_values)), dx=0.01) + self.x_start

        return pos_F,der
        

class Elastomer0():
    def __init__(self,delta_t=0.01,t0=1,F0=1,k1=0.1,k2=0.167,xa=1.5*100,xb=6*100):
        self.F0=F0
        self.t0=t0
        self.delta_t=delta_t
        self.EPR=0 #Equilibrium position ratio
        # 示例数据点
        # x1 = np.array([1, 3, 4, 5, 10,15])
        # y1 = np.array([0, 0.45,0.65,0.8,1.05,1.15])
        # x2 = np.array([1, 5])
        # y2 = np.array([1.6, 1.8])
        x1 = np.array([0,xb*k1,xb*k1+1])
        y1 = np.array([0, xb,xb])
        x2 = np.array([0, (xb-xa)*k2,(xb-xa)*k2+1])
        y2 = np.array([xa, xb,xb])
        # 创建插值函数
        self.interpolation_func1 = smooth_curve(x1, y1)
        self.interpolation_func2 = smooth_curve(x2, y2)
    def _transient_line(self,F):
        # pos=115-115*math.exp(-(F-1)/2.62)
        pos=self.interpolation_func1(F)
        der=derivative(self.interpolation_func1, F, dx=1e-6)
        return pos,der
    def _steady_line(self,F):
        pos=self.interpolation_func2(F)
        der=derivative(self.interpolation_func2, F, dx=1e-6)
        return pos,der
    def _final_EPR(self,F):
        epr=1-math.exp(-F/self.F0)
        return epr
    def step(self,F):
        pos1,der1=self._transient_line(F)
        pos2,der2=self._steady_line(F)
        self.EPR+=-self.delta_t/self.t0*(self.EPR-self._final_EPR(F))
        pos=(pos2-pos1)*self.EPR+pos1
        der=(der2-der1)*self.EPR+der1
        return pos,der*2.2
    def reset(self,delta_t=0.01,t0=1,F0=1):
        self.F0=F0
        self.t0=t0
        self.delta_t=delta_t
        self.EPR=0

class Elastomer1():
    def __init__(self,delta_t=0.01):
        self.f_p_sim=F_Pos_simulator(delta_t=delta_t)
        self.f_p_sim.generate()
    def step(self,F):#ma=F-k*x-mu*v
        x,k_inverse=self.f_p_sim.pos_F_func(F)
        self.f_p_sim.step()
        return x,k_inverse
    def reset(self):
        self.f_p_sim.generate()

if __name__=='__main__':
    s=F_Pos_simulator()
    s.generate()
    x_new = np.linspace(0, 5, 100)
    y_new = np.linspace(0, 5, 100)
    for i in range(100):
        y_new[i]=s.pos_F_func(x_new[i])[0]
    plt.figure(figsize=(4, 4))
    plt.plot(y_new, x_new)
    plt.legend()
    plt.xlabel('x(mm)')
    plt.ylabel('F_N(N)')
    plt.title('Instantaneous Force-Deformation Curve')
    plt.show()    
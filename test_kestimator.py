from k_simulator import Elastomer0
from k_simulator import Elastomer1
from rnd_generator import Generater
from kEstimator_by_czy_usedinlstm import DexHand_kEstimator
from plot import kAssessor
from plot import evaluate
import math
import numpy as np
import time
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径             

if __name__=='__main__':
    def Fn(t): #t/s
        if 0<=t<5:
            return t/5*(1-0.1)+0.1
        elif 5<=t<7:
            return (t-5)/2*(5-1)+1
        elif 7<=t<9:
            return (9-t)/2*(5-1)+1
        elif 9<=t<11:
            return (t-9)/2*(5-1)+1
        elif 11<=t<16:
            return 5
        elif 16<=t<18:
            return (18-t)/2*(5-1)+1
        elif 18<=t<23:
            return 1
        else:
            return 0
    def Fs(t):
        return 2.5-2.4*math.cos(t*1)
    
    kestimator=DexHand_kEstimator()
    F=Generater()
    kassessor=kAssessor()
    F_error=[]
    k_lmse=[]
    k_rdse=[]
    round=1
    for i in range(round):
        kestimator.reset()
        cnt=0
        # np.random.seed(3649601111)
        FN=F.generate_func()
        hand,balloon=F.generate_elas()
        # hand=Elastomer0(delta_t=0.01,t0=1,F0=1,k1=0.1,k2=0.167,xa=1.5*100,xb=3*100)
        # balloon=Elastomer0(delta_t=0.01,t0=1,F0=1,k1=0.002,k2=0.002,xa=0*100,xb=10*100)   
        # balloon=Elastomer1(delta_t=0.01,k=0.001,x0=1000,m=1e-7) 
        k_reals=[]
        time_values=[]
        k_ests=[]
        pos_values=[]
        F_Ns=[]
        with open(curr_path+'/output.txt','w') as f:
            while(1):
                F_N=FN(cnt*0.01)
                if cnt*0.01>=20:
                    break

                pos1,der_real1=hand.step(F_N)
                pos2,der_real2=balloon.step(F_N)
                pos1=0
                der_real1=0
                pos=pos1+pos2
                if der_real1+der_real2 ==0:
                    k_real=0.8
                else:  
                    k_real=1/(der_real1+der_real2)
                kestimator.data_push_in(pos,F_N,F_N)

                k_reals.append(k_real)
                k_ests.append(kestimator.k)
                time_values.append(cnt*0.01)
                pos_values.append(pos/100)
                cnt+=1
                F_Ns.append(F_N)
                print(F_N,F_N,k_real,kestimator.k_out,pos/100,file=f)
        F_error1,k_lmse1,k_rdse1=evaluate(plot=True,time_values=time_values,F_real_values=F_Ns,F_goal_values=F_Ns,k_real_values=k_reals,k_est_values=k_ests,pos_values=pos_values)
        F_error.append(F_error1)
        k_lmse.append(k_lmse1)
        k_rdse.append(k_rdse1)
    print(round,'轮平均F:',np.mean(F_error),'k_LMSE:',np.mean(k_lmse),'k_RDSE:',np.mean(k_rdse)) 
    print(round,'轮方差F:',np.var(F_error),'k_LMSE:',np.var(k_lmse),'k_RDSE:',np.var(k_rdse)) 

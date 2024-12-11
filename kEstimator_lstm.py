import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from train_test import StiffnessEstimatorRNN
from train_test import load_data
import kEstimator_by_czy_usedinlstm
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径   

class DexHand_kEstimator():
    def __init__(self):
        self.input_size = 3 # 输入特征的维度
        self.hidden_size = 512 # 隐藏层的维度
        self.output_size = 1 # 输出特征的维度
        self.num_layers = 5
        model_load_path=curr_path+'/models/model512_5_rnd2_rdseloss_freeab_10000data.pt'
        data_path=curr_path+'/data/data_rnd2'
        self.manualkEstimator=kEstimator_by_czy_usedinlstm.DexHand_kEstimator()
        data,labels,self.input_mean,self.input_std=load_data(l=0,r=1,save_path=data_path)
        self.model = StiffnessEstimatorRNN(self.input_size, self.hidden_size, self.output_size,self.num_layers)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        
        self.model.to(self.device)
        print('device:',self.device)

        self.h_t = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        self.c_t = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        self.k=None

    def data_push_in(self,nowpos,nowforce,goalforce):
        self.manualkEstimator.data_push_in(nowpos,nowforce,goalforce)
        input=(np.array([nowpos,goalforce,self.manualkEstimator.k])-self.input_mean)/self.input_std
        input_t=torch.Tensor(np.array([input])).to(self.device)
        # 对每一层进行LSTMCell前向传播
        for layer in range(self.num_layers):
            self.h_t[layer], self.c_t[layer] = self.model.lstm_cells[layer](input_t, (self.h_t[layer], self.c_t[layer]))
            # 添加激活函数
            self.h_t[layer] = torch.tanh(self.h_t[layer])

            # 更新输入，当前层的输出作为下一层的输入
            input_t = self.h_t[layer]
        output_t=self.model.activation(self.model.fc(self.h_t[-1]))*3*(0.0005+torch.relu((self.model.a.to(self.device)*(torch.Tensor(np.array([input[-1]])).to(self.device))+self.model.b.to(self.device))))
        if(not self.k):
            self.k=0.05
        else:
            self.k=output_t.item()

        if self.k<0.0005:
            self.k = 0.0005
        elif self.k>0.8:
            self.k=0.8

        self.k_out=self.k
        # self.k = 0.05
    def reset(self):
        self.h_t = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        self.c_t = [torch.zeros(1, self.hidden_size).to(self.device) for _ in range(self.num_layers)]
        self.k=None     
        self.manualkEstimator.reset()   
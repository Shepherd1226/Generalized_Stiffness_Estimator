import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from kEstimator import DexHand_kEstimator
from k_simulator import Elastomer0
from k_simulator import Elastomer1
from rnd_generator import Generater
from plot import kAssessor
from plot import evaluate
import random
import math
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径   

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
         
def generate_data(num_samples=100, sequence_length=2000,save=True,save_path=curr_path+'/data'):
    F=Generater()
    input = np.empty((num_samples, sequence_length, 3))
    target = np.empty((num_samples, sequence_length, 1))

    for num_sample in range(num_samples):
        kestimator=DexHand_kEstimator()
        hand, balloon = F.generate_elas()
        # hand=Elastomer0(delta_t=0.01,t0=1,F0=1,k1=0.1,k2=0.167,xa=1.5*100,xb=3*100)
        # balloon=Elastomer0(delta_t=0.01,t0=1,F0=1,k1=0.002,k2=0.002,xa=0*100,xb=10*100)   
        # balloon=Elastomer1(delta_t=0.01,k=0.001,x0=1000,m=1e-7) 
        FN = F.generate_func()

        input_per_epoch = np.empty((sequence_length, 3))
        target_per_epoch = np.empty((sequence_length, 1))

        for i in range(sequence_length):
            if (i + 1) * 0.01 > 20:
                print('too many steps per eps!')

            F_N = FN((i + 1) * 0.01)

            pos1, der_real1 = hand.step(F_N)
            pos2, der_real2 = balloon.step(F_N)
            pos = pos1 + pos2
            if der_real1 + der_real2 <= 1 / 0.8:
                k_real = 0.8
            else:
                k_real = 1 / (der_real1 + der_real2)

            kestimator.data_push_in(pos,F_N,F_N)

            target_per_epoch[i] = k_real
            input_per_epoch[i] = np.array([pos, F_N,kestimator.k])

        input[num_sample] = input_per_epoch
        target[num_sample] = target_per_epoch

        if (num_sample+1)%10==0 or num_samples==num_sample+1:
            print(num_sample+1,'of',num_samples,'data generated')
    
    input_mean = np.mean(input, axis=(0, 1))
    input_std = np.std(input, axis=(0, 1))
    input =  (input - input_mean) / input_std
    if save:
        np.save(save_path+'/data.npy', input)
        np.save(save_path+'/labels.npy', target)
        np.save(save_path+'/input_mean.npy', input_mean)
        np.save(save_path+'/input_std.npy', input_std)
    return input,target,input_mean,input_std

def load_data(l=0,r=100,save_path=curr_path+'/data'):
    data=np.load(save_path+'/data.npy')
    labels=np.load(save_path+'/labels.npy')
    input_mean=np.load(save_path+'/input_mean.npy')
    input_std=np.load(save_path+'/input_std.npy')
    data=data[l:r,:,:]
    labels=labels[l:r,:,:]
    return data,labels,input_mean,input_std

class RDSELoss(nn.Module):#RDSELoss relative double square error
    def __init__(self):
        super(RDSELoss, self).__init__()

    def forward(self, a,b):
        x=a/b
        loss = torch.mean(x**2 + 1/x**2 - 2)
        return loss

class LMSELoss(nn.Module):
    def __init__(self):
        super(LMSELoss, self).__init__()

    def forward(self, a,b):
        loss = nn.functional.mse_loss(torch.log(a),torch.log(b))
        return loss
    
class StiffnessEstimatorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(StiffnessEstimatorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 多层LSTMCell
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size)])
        for _ in range(1, num_layers):
            self.lstm_cells.append(nn.LSTMCell(hidden_size, hidden_size))

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()  # 添加激活函数
        self.a=nn.Parameter(torch.tensor(3.0944e-2),requires_grad=True)
        self.b=nn.Parameter(torch.tensor(1.7567e-2),requires_grad=True)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        # 初始化每一层的隐藏状态和细胞状态
        h_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        outputs = []

        # 逐时间步进行前向传播
        for step in range(sequence_length):
            # 获取当前时间步的输入
            input_t = x[:, step, :]

            # 对每一层进行LSTMCell前向传播
            for layer in range(self.num_layers):
                h_t[layer], c_t[layer] = self.lstm_cells[layer](input_t, (h_t[layer], c_t[layer]))
                # 添加激活函数
                h_t[layer] = torch.tanh(h_t[layer])

                # 更新输入，当前层的输出作为下一层的输入
                input_t = h_t[layer]

            # 保存最后一层的输出self.activation(self.fc(h_t[-1]))*3
            outputs.append(self.activation(self.fc(h_t[-1]))*3*(0.0005+torch.relu(self.a.repeat(batch_size, 1).to(device)*x[:, step, -1:]+self.b.repeat(batch_size, 1).to(device))))

        # 全连接层和激活函数
        outputs = torch.stack(outputs, dim=1)  # 在时间维度上堆叠
        return outputs
    
def train(model, dataloader, criterion, optimizer, epochs=50,save_path=curr_path+'/models/model.pt'):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        cnt=0
        for inputs, targets in dataloader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs[:,1:,:],targets[:,1:,:])

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt+=1
            # print('Epoch',epoch+1,'round',cnt)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')
        if (epoch + 1) % 1 == 0:  # 每1个回合保存一次
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)   

def test(model,round=2,plot=True,trained_data=False,l=0,r=100,load_path=curr_path+'/data/data_fixed'):
    model.eval()
    F_error=0
    k_lmse=0
    k_rdse=0
    for i in range(round):
        seq_len=2000
        if trained_data:
            x=random.randint(l,r-1)
            data,labels,data_mean0,data_std0=load_data(l=x,r=x+1,save_path=load_path)
            data_mean1,data_std1=data_mean0,data_std0
        else:
            data,labels,data_mean1,data_std1=generate_data(num_samples=1,sequence_length=seq_len,save=False)
            rr,rrr,data_mean0,data_std0=load_data(l=0,r=1,save_path=load_path)
        data=((data[0]*data_std1+data_mean1)-data_mean0)/data_std0
        labels=labels[0]
        pos=[]
        F_N=[]
        k_ests=[]
        k_reals=[]
        time_values=[]
        h_t = [torch.zeros(1, hidden_size).to(device) for _ in range(num_layers)]
        c_t = [torch.zeros(1, hidden_size).to(device) for _ in range(num_layers)]
        with open(curr_path+'/output.txt','w') as f:
            for t in range(seq_len):
                input_t=torch.Tensor(data[t:t+1, :]).to(device)
                # 对每一层进行LSTMCell前向传播
                for layer in range(num_layers):
                    h_t[layer], c_t[layer] = model.lstm_cells[layer](input_t, (h_t[layer], c_t[layer]))
                    # 添加激活函数
                    h_t[layer] = torch.tanh(h_t[layer])

                    # 更新输入，当前层的输出作为下一层的输入
                    input_t = h_t[layer]
                # output_t=model.activation(model.fc(h_t[-1])*(torch.Tensor(data[t:t+1, -1]).to(device)))/3
                # model.activation(model.fc(h_t[-1]))*3
                output_t=model.activation(model.fc(h_t[-1]))*3*(0.0005+torch.relu((model.a.to(device)*(torch.Tensor(data[t:t+1, -1]).to(device))+model.b.to(device))))
                if(t==0):
                    k_ests.append(0.05)  
                else:
                    k_ests.append(output_t.item())    
                k_reals.append(labels[t][0])
                input_original=data[t]*data_std0+data_mean0
                pos.append(input_original[0]/100)
                F_N.append(input_original[1])
                time_values.append(len(time_values) * 0.01)
                print(F_N[-1],F_N[-1],k_reals[-1],k_ests[-1],pos[-1],file=f)
        F_error1,k_lmse1,k_rdse1=evaluate(time_values=time_values,F_real_values=F_N,F_goal_values=F_N,k_real_values=k_reals,k_est_values=k_ests,pos_values=pos,plot=plot)
        F_error+=F_error1
        k_lmse+=k_lmse1
        k_rdse+=k_rdse1
    F_error/=round
    k_lmse/=round
    k_rdse/=round
    print(round,'轮平均F:',F_error,'k_LMSE:',k_lmse,'k_RDSE:',k_rdse)

if __name__=='__main__':
    input_size = 3 # 输入特征的维度
    hidden_size = 512 # 隐藏层的维度
    output_size = 1 # 输出特征的维度
    num_layers = 5
    batch_size = 24
    data_path=curr_path+'/data/data_rnd2'
    model_load_path=curr_path+'/models/model512_5_rnd2_rdseloss_freeab_10000data.pt'
    model_save_path=curr_path+'/models/model512_5_rnd2_rdseloss_freeab_10000data.pt'

    # generate_data(num_samples=10000,sequence_length=2000,save_path=data_path)
    data,labels,input_mean,input_std=load_data(l=0,r=10000,save_path=data_path)
    # 将数据转换为 PyTorch 的 Tensor
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    # 创建 DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("data loaded successfully.")

    # 实例化模型、损失函数和优化器
    model = StiffnessEstimatorRNN(input_size, hidden_size, output_size,num_layers)
    criterion = RDSELoss()
    # criterion = LMSELoss()
    # criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 检查是否可用gpu加速
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print('device:',device)

    # 加载已有模型
    checkpoint = torch.load(model_load_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model and optimizer loaded successfully.")

    # 调用训练函数进行模型训练
    train(model, dataloader, criterion, optimizer, epochs=1000,save_path=model_save_path)

    # 调用测试函数进行测试
    test(model=model,round=10,plot=True,trained_data=False,l=0,r=10000,load_path=data_path)




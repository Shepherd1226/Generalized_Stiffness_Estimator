import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from kEstimator_by_czy_usedinlstm import DexHand_kEstimator
from rnd_generator import Generater
import math
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
        FN = F.generate_func()

        input_per_epoch = np.empty((sequence_length, 3))
        target_per_epoch = np.empty((sequence_length, 1))

        for i in range(sequence_length):
            if (i + 1) * 0.01 > 20:
                print('too many steps per eps!')

            F_N = FN((i + 1) * 0.01)

            # pos1, der_real1 = hand.step(F_N)
            pos, der_real = balloon.step(F_N)
            k_real = 1 / der_real

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
    
class StiffnessEstimatorRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(StiffnessEstimatorRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Multi-layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()  # Activation function
        self.a = nn.Parameter(torch.tensor(3.0944e-2), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.7567e-2), requires_grad=True)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply the fully connected layer and activation function to all time steps
        fc_out = self.fc(lstm_out)
        activated = self.activation(fc_out)

        # Apply the scaling and additional transformation
        combined = activated * 3 * (torch.relu(self.a * x[:, :, -1:] + self.b))

        return combined

def train(model, dataloader, criterion, optimizer, epochs=50,save_path=curr_path+'/models/model.pt'):
    model.train()
    total_loss = 0.0
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

        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')
        if (epoch + 1) % 1 == 0:  # 每1个回合保存一次
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)  
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs[:, 1:, :], targets[:, 1:, :])

            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__=='__main__':
    input_size = 3 # 输入特征的维度
    hidden_size = 512 # 隐藏层的维度
    output_size = 1 # 输出特征的维度
    num_layers = 5
    batch_size = 32
    data_path = curr_path + '/data/data_rnd_essay'
    model_load_path = curr_path + '/models/model512_5_model2_10000data.pt'
    model_save_path = curr_path + '/models/model512_5_model2_10000data.pt'

    # generate_data(num_samples=10000, sequence_length=2000, save_path=data_path)
    data, labels, input_mean, input_std = load_data(l=2000, r=10000, save_path=data_path)
    # 将数据转换为 PyTorch 的 Tensor
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    # 创建 DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("data loaded successfully.")

    # 创建验证集 DataLoader
    val_data, val_labels, _, _ = load_data(l=0, r=2000, save_path=data_path)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    val_dataset = TensorDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("validation data loaded successfully.")

    # 实例化模型、损失函数和优化器
    model = StiffnessEstimatorRNN2(input_size, hidden_size, output_size, num_layers)
    criterion = RDSELoss()
    # criterion = LMSELoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 检查是否可用gpu加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print('device:', device)

    # 加载已有模型
    checkpoint = torch.load(model_load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model and optimizer loaded successfully.")

    with open(curr_path + '/train_validate_figure.txt', 'w') as a:
        for i in range(169):
            # 调用训练函数进行模型训练
            train_loss = train(model, dataloader, criterion, optimizer, epochs=1, save_path=model_save_path)

            # 调用测试函数进行验证
            validate_loss = validate(model=model, dataloader=val_dataloader, criterion=criterion)
            print(train_loss, validate_loss, file=a)
            print('Epoch', i + 1, 'train_loss:', train_loss, 'validate_loss:', validate_loss)





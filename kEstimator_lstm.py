import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from train_test import StiffnessEstimatorRNN2  # Updated to use the new class
from train_test import load_data
import kEstimator_by_czy_usedinlstm
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径   

class DexHand_kEstimator():
    def __init__(self):
        self.input_size = 3  # 输入特征的维度
        self.hidden_size = 512  # 隐藏层的维度
        self.output_size = 1  # 输出特征的维度
        self.num_layers = 5
        model_load_path = curr_path + '/models/model512_5_model2_10000data.pt'
        data_path = curr_path + '/data/data_rnd_essay'
        self.manualkEstimator = kEstimator_by_czy_usedinlstm.DexHand_kEstimator()
        data, labels, self.input_mean, self.input_std = load_data(l=0, r=1, save_path=data_path)
        self.model = StiffnessEstimatorRNN2(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)
        print('device:', self.device)

        # Initialize hidden and cell states for LSTM
        self.hidden_state = None
        self.k = None

    def data_push_in(self, nowpos, nowforce, goalforce):
        self.manualkEstimator.data_push_in(nowpos, nowforce, goalforce)
        input_vector = (np.array([nowpos, goalforce, self.manualkEstimator.k]) - self.input_mean) / self.input_std
        input_t = torch.Tensor(np.array([[input_vector]])).to(self.device)  # Shape: (1, 1, input_size)

        # Forward pass through LSTM
        if self.hidden_state is None:
            lstm_out, self.hidden_state = self.model.lstm(input_t)  # Initialize hidden state
        else:
            lstm_out, self.hidden_state = self.model.lstm(input_t, self.hidden_state)  # Use existing hidden state

        # Final output computation
        output_t = self.model.activation(self.model.fc(lstm_out[:, -1, :])) * 3 * (
            torch.relu(self.model.a * input_vector[-1] + self.model.b))

        # Update k value
        if self.k is None:
            self.k = 0.05
        else:
            self.k = output_t.item()

        if self.k < 0.0005:
            self.k = 0.0005
        elif self.k > 0.8:
            self.k = 0.8

        self.k_out = self.k

    def reset(self):
        # Reset hidden state and k value
        self.hidden_state = None
        self.k = None     
        self.manualkEstimator.reset()

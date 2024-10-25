import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径

# Load the data
epochs = []
train_loss = []
val_loss = []

data_file = curr_path+'/train_validate_figure.txt'
with open(data_file, "r") as file:
    for i, line in enumerate(file):
        epoch_train_loss, epoch_val_loss = map(float, line.strip().split())
        epochs.append(i + 1)  # Epochs start from 1
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

# Plot the data
plt.figure(figsize=(5, 4))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 5)  # Limit y-axis from 0 to 5
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

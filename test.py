import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from models.model import *
from data_loader import *

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# 加载数据集
dataset = sio.loadmat('./data/dataset_28.mat')
full_dataset = LoadDataset(dataset)
test_size = int(len(full_dataset) * 0.2)
train_size = len(full_dataset) - test_size

_, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], torch.manual_seed(0))
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

# 加载模型
model = torch.load('checkpoints/best_author.pth')
model = model.to(device)
model.eval()

# 定义损失函数
loss = nn.L1Loss()
loss = loss.to(device)

# 测试函数
def test_accuracy(model, test_loader, loss):
    model.eval()
    total_loss = 0
    pred_list, target_list = np.zeros((1, 2)), np.zeros((1, 2))
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            Data = data["feature"].to(device)
            target = data["gt"].to(device)
            output = model(Data)

            # 累计损失
            loss_curr = loss(output, target).item()
            total_loss += loss_curr

            # 收集预测和标签
            pred_list = np.append(pred_list, output.cpu().numpy(), axis=0)
            target_list = np.append(target_list, target.cpu().numpy(), axis=0)

    # 计算预测和真实标签之间的 RMSE
    rmse = np.sqrt(np.mean((pred_list[1:] - target_list[1:]) ** 2))
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, RMSE: {rmse:.4f}")
    return rmse

# 测试模型精度
rmse = test_accuracy(model, test_loader, loss)
print(f"Final Test Accuracy (RMSE): {rmse:.4f}")
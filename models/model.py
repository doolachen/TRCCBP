import torch.nn as nn
from models.blocks import *
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = CNN(kernel_size=3)
        self.BN = nn.BatchNorm1d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.drop = nn.Dropout(0.2)
        self.conv_block2 = CNN(kernel_size=5)
        self.gru = GRU()
        self.multihead_attention = MultiHead_Attention(in_ch=128, num_head=8)
        # self.fc1 = FC(28*128, 100)
        # fc1 延迟初始化
        self.fc1 = None 
        self.fc2 = FC(100, 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_block1(x)    # 卷积模块 1
        x2 = self.conv_block2(x)    # 卷积模块 2
        x3 = torch.cat([x1,x2],dim=1)   # 按通道维度拼接
        x3 = x3.transpose(1, 2) # 1 2 换一下维度 适配 GRU
        x3 = self.gru(x3)   # GRU
        x3 = self.multihead_attention(x3)   # 多头注意力
        x3 = x3.transpose(1, 2) # 调回来
       
        if self.fc1 is None:
            in_features = x3.shape[1] * x3.shape[2]  # 计算输入特征维度
            self.fc1 = nn.Linear(in_features, 100).to(x3.device)  # 动态初始化全连接层
            
        x3 = x3.flatten(start_dim=1)  # 展平成 (batch_size, in_features)
        x3 = self.fc1(x3)   # 第一个全连接层
        x3 = self.act(x3)   # 激活函数
        out = self.fc2(x3)  # 输出层
        return out


if __name__ == '__main__':
    data1 = torch.randn([64, 1, 30])
    # print(type(data1))
    # data2 = torch.randn([32, 50, 1, 3])
    model = Net()
    out = model(data1)
    print(out.shape)

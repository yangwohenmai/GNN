# https://www.cnblogs.com/codingbao/p/17763312.html
# https://aistudio.baidu.com/datasetdetail/179508
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import numpy as np
import baostock as bs

import pandas as pd
import 构建图train数据
import 获取股市码表
import 获取个股票行情

#from torch_geometric.data.collate import collate
from torch_geometric.data import InMemoryDataset


构建图train数据
lg = bs.login()
stockPriceDic = 获取个股票行情.GetStockPriceDWMBaostock('600000.SH', 0)
data_list = 构建图train数据.TrainDataInt(stockPriceDic)
dataset, slices = InMemoryDataset.collate(data_list)
print(dataset)







import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling,SAGEConv
#载入数据
#dataset = Planetoid(root='./data/test11', name='Cora')
#data = dataset[0]
#print(data.x[0].tolist())
#print(data.y.tolist())
#print(data.edge_index[0].tolist())
#print(data.edge_index[1].tolist())
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(128, 64)
        self.conv3 = SAGEConv(64, 16)
        self.conv4 = GCNConv(64, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(data.x)
print(data.y)
#模型训练
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    #loss = F.nll_loss(out, data.y)   #损失仅仅计算的是训练集的损失
    loss.backward()
    optimizer.step()
#测试：
model.eval()
test_predict = model(data.x, data.edge_index)[data.test_mask]
max_index = torch.argmax(test_predict, dim=1)
test_true = data.y[data.test_mask]
correct = 0
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
print('测试集准确率为：{}%'.format(correct*100/len(test_true)))


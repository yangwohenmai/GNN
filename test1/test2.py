#!/usr/bin/env python
# coding: utf-8

# In[5]:


#https://blog.csdn.net/steven_ysh/article/details/128383257
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

# 简易可视化函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=41), with_labels=True, node_color=color, cmap="Set2")
    plt.show()
    
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4], [1, 2, 4, 0, 3, 4, 2]], dtype=torch.long)  # 第一组为边起点，第二组为边终点
x = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float)        # 节点特征，这里使用一维特征
data = Data(x=x, edge_index=edge_index)                               # 构建 Data 实例
G = to_networkx(data=data)                                            # 将 Data 实例转换到 networkx.Graph
visualize_graph(G=G, color='pink')                                    # 可视化构建的图


# In[10]:


from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 简易可视化函数
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()
    
dataset = KarateClub()
print(f'Dataset: {dataset}')                                # 数据集名称，KarateClub()
print('='*20)
print(f'Number of graphs: {len(dataset)}')                  # 图数量，1
print(f'Number of features: {dataset.num_features}')        # 点特征数量，34
print(f'Number of classes: {dataset.num_classes}')          # 数据类别数量，4

data = dataset[0]                                           # 获取 dataset 中的一个数据（图）
print(f'An undirected graph: {data.is_undirected()}')       # 是否无向图，True
print('='*30)
print(f'Num of edges: {data.num_edges}')                    # 边数量，156=78x2（因为无向图，所以乘2）
print(f'Num of nodes: {data.num_nodes}')                    # 点数量，34
print(f'Num of node features: {data.num_node_features}')    # 点特征数量，34
print(f'Shape of x: {np.shape(data.x)}')                    # 节点特征矩阵，[34, 34]
print(f'Shape of edge_index: {np.shape(data.edge_index)}')  # 边的起终点矩阵，[2, 156]
print(f'Attr of edge: {data.edge_attr}')                    # 边特征矩阵，None
print(f'Shape of y: {np.shape(data.y)}')                    # 真值，[34]（对应 34 个点）
print(f'Pos of data: {data.pos}')                           # 节点位置矩阵，None

G = to_networkx(data, to_undirected=True)                   # 将 Data 实例转换到 networkx.Graph
visualize_graph(G, data.y)                                  # 可视化构建的图


# In[8]:


from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

# 下载数据集，数据存放路径为当前路径下 data/ENZYMES 文件夹
dataset = TUDataset(root='data/ENZYMES', name='ENZYMES', use_node_attr=True)
# 加载数据集，使用的 batch size 为 32，且打乱顺序
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 获取每一个 batch
for batch in loader:
    print(batch)  # DataBatch(edge_index=[2, 4036], x=[1016, 21], y=[32], batch=[1016], ptr=[33])
    print(batch.num_graphs)                        # 32
    x = scatter_mean(batch.x, batch.batch, dim=0)  # 分别为每个图的节点维度计算平均的节点特征
    print(x.size())                                # torch.Size([32, 21])
    break                                          # 迭代一次后退出


# In[9]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 下载并加载数据集
dataset = Planetoid(root='data/Planetoid', name='Cora')

# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 模型相关配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
model = GCN(hidden_channels=16).to(device)
data = dataset[0].to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练函数，返回损失值
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# 测试函数，返回准确率
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

# 训练 200 轮
for epoch in range(1, 201):
    loss = train()
print(f'Train loss: {loss:.4f}')         # Train loss: 0.0223

test_acc = test()
print(f'Test accuracy: {test_acc:.4f}')  # Test accuracy: 0.8070


# In[ ]:





# In[ ]:





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

embed_dim = 128
num_embeddings = 900
from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
class Net(torch.nn.Module): #针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层 
        self.conv1 = SAGEConv(embed_dim, 128)
        # 定义池化层
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        # 定义嵌入层 num_embeddings数量比词类型个数至少多1 
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        #self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +10, embedding_dim=embed_dim)
        # 定义线性层
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        # 定义标准化层和激活函数
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch # x:n*1,其中每个图里点的个数是不同的
        #print(x)
        # 使用嵌入层将节点特征进行编码。这里的item_embedding是一个嵌入层，用于将节点的特征映射到一个低维空间。
        x = self.item_embedding(x)# n*1*128 特征编码后的结果
        #print('item_embedding',x.shape)
        # 去掉特征的维度为1的维度，使特征具有形状n * 128。
        x = x.squeeze(1) # n*128        
        #print('squeeze',x.shape)
        a = self.conv1(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))# n*128
        #print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)# pool之后得到 n*0.8个点
        #print('self.pool1',x.shape)
        #print('self.pool1',edge_index)
        #print('self.pool1',batch)
        #x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # 全局平均池化
        x1 = gap(x, batch)
        #print('gmp',gmp(x, batch).shape) # batch*128
        #print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        #print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #print('pool2',x.shape)
        #print('pool2',edge_index)
        #print('pool2',batch)
        #x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # 全局特征
        x2 = gap(x, batch)
        #print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        #print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #print('pool3',x.shape)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        #print('x3',x3.shape)# batch * 256
        # 将三个尺度的全局特征相加，以获取不同尺度的全局信息。
        x = x1 + x2 + x3 # 获取不同尺度的全局特征
 
        x = self.lin1(x)
        #print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        #print('lin2',x.shape)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        # 最后一层使用了Sigmoid激活函数，将模型的输出压缩到0到1之间，用于进行二元分类。
        # .squeeze(1)操作将结果的维度从(batch_size, 1)变为(batch_size,)。
        x = torch.sigmoid(self.lin3(x)).squeeze(1)#batch个结果
        #print('sigmoid',x.shape)
        return x


from torch_geometric.loader import DataLoader

def train():
    model.train()
 
    loss_all = 0
    for data in train_loader:
        data = data
        #print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)
    
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
#train_loader = DataLoader(dataset, batch_size=64)
train_loader = data_list
for epoch in range(10):
    print('epoch:',epoch)
    loss = train()
    print(loss)



from  sklearn.metrics import roc_auc_score

def evalute(loader,model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data#.to(device)
            pred = model(data)#.detach().cpu().numpy()

            label = data.y#.detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction =  np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels,prediction) 


for epoch in range(1):
    roc_auc_score = evalute(dataset,model)
    print('roc_auc_score',roc_auc_score)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import baostock as bs
import os
import sys
# sys.path.append用于向环境变量中添加路径
#sys.path.append('..\..')
# 打印文件绝对路径（absolute path）
#print (os.path.abspath(__file__))  
# 打印文件父目录的父目录的路径（文件的上两层目录）
#print (os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))) 
# 要调取其他目录下的文件。 需要在atm这一层才可以
#BASE_DIR=  os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#print(BASE_DIR)
# 将这个路径添加到环境变量中。
#sys.path.append(BASE_DIR)
# 打印当前环境变量包含的所有路径
#print(sys.path)

sys.path.append('..')
from Data import StockPool
from Data import StockData
from Data import TrainData


lg = bs.login()
#stockPoolList = StockPool.GetStockPool('',False,'')
#for code in StockPool.GetALLStockListBaostock().keys():
stockPriceDic = StockData.GetStockPriceDWMBaostock('600000.SH', 0)
data = TrainData.TrainDataInt(stockPriceDic)[0]
split = int(len(stockPriceDic)*0.8)
train_mask=[]
test_mask=[]
for i in range(0,len(stockPriceDic)):
    if i < split:
        train_mask.append(True)
        test_mask.append(False)
    else:
        train_mask.append(False)
        test_mask.append(True)

#print(data.x)
#print(data.y)
#print(data.x[0].tolist())
#print(data.y.tolist())
#print(data.edge_index[0].tolist())
#print(data.edge_index[1].tolist())
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(dataset.num_features, 128)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv1 = GCNConv(6, 32)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 2)
        self.conv5 = GCNConv(256, 2)
        #self.conv4 = GCNConv(64, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.sigmoid(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

#模型训练
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(data.x.to(torch.float32), data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask])   #损失仅仅计算的是训练集的损失
    loss.backward()
    optimizer.step()
#测试：
model.eval()
#test_predict = model(data.x, data.edge_index)[data.test_mask]
test_predict = model(data.x.to(torch.float32), data.edge_index)[test_mask]
max_index = torch.argmax(test_predict, dim=1)
#test_true = data.y[data.test_mask]
test_true = data.y.to(torch.long)[test_mask]
correct = 0
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
print('测试集准确率为：{}%'.format(correct*100/len(test_true)))




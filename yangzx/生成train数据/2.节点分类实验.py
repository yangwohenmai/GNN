import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



import 构建图train数据_ForInMemoryDataset
import baostock as bs
import 获取个股票行情
#载入数据
#Cora数据集包含2708篇科学出版物， 5429条边，总共7种类别。从2708篇论文中统计出1433个高频词。
#每篇论文x是一个[1,1433]的one-hot词向量。对应的y是0-7。去重后的边向量为[2.1056]


lg = bs.login()
stockPriceDic = 获取个股票行情.GetStockPriceDWMBaostock('600000.SH', 0)
data = 构建图train数据_ForInMemoryDataset.TrainDataInt(stockPriceDic)[0]
#print(data.x[0].tolist())
#print(data.y.tolist())
print(data.edge_index[0].tolist())#[2,10556]
print(data.edge_index[1].tolist())
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(dataset.num_features, 128)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv1 = GCNConv(4, 16)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, 2)
        #self.conv4 = GCNConv(64, dataset.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.sigmoid(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(data.x)#[2708,1433] onehot
print(data.y)#[2708] 7种类型
#模型训练
model.train()
for epoch in range(400):
    optimizer.zero_grad()
    out = model(data.x.to(torch.float32), data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    loss = F.nll_loss(out, data.y.to(torch.long))   #损失仅仅计算的是训练集的损失
    loss.backward()
    optimizer.step()
#测试：
model.eval()
#test_predict = model(data.x, data.edge_index)[data.test_mask]
test_predict = model(data.x.to(torch.float32), data.edge_index)
max_index = torch.argmax(test_predict, dim=1)
#test_true = data.y[data.test_mask]
test_true = data.y.to(torch.long)
correct = 0
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
print('测试集准确率为：{}%'.format(correct*100/len(test_true)))




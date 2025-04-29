import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import baostock as bs
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_score
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
#画图参考资料
#https://zhuanlan.zhihu.com/p/634602384?utm_id=0

sys.path.append('..')
from DataBase import StockPool
from DataBase import StockData
from DataBase import TrainData


lg = bs.login()
#stockPoolList = StockPool.GetStockPool('',False,'')
#for code in StockPool.GetALLStockListBaostock().keys():
stockPriceDic = StockData.GetStockPriceDWMBaostock('600000.SH', 0)
data = TrainData.TrainDataInt(stockPriceDic)[0]
split_train = int(len(stockPriceDic)*0.75)
split_val = int(len(stockPriceDic)*0.85)
train_mask=[]
test_mask=[]
val_mask=[]
for i in range(0,len(stockPriceDic)):
    if i < split_train:
        train_mask.append(True)
        test_mask.append(False)
        val_mask.append(False)
    elif i>=split_train and i<split_val:
        train_mask.append(False)
        val_mask.append(True)
        test_mask.append(False)
    else:
        train_mask.append(False)
        test_mask.append(True)
        val_mask.append(False)


def plot_metrics(precisions, recalls, f1s, losses):
    """
    训练指标变化过程可视化
    :param precisions:
    :param recalls:
    :param f1s:
    :param losses:
    :return:
    """
    epochs = range(1, len(precisions) + 1)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, precisions, 'g', label='Precision')
    plt.plot(epochs, recalls, 'r', label='Recall')
    plt.plot(epochs, f1s, 'm', label='F1')
    plt.plot(epochs, losses, 'b', label='Loss')
    plt.title('Training And Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

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
        self.conv1 = GCNConv(7, 32)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GATConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GATConv(128, 2)
        self.conv5 = GATConv(256, 2)
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

# 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
precisions, recalls, f1s, losses = [], [], [], []
#模型训练/验证
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(data.x.to(torch.float32), data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask])   #损失仅仅计算的是训练集的损失
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    #启用验证模式
    model.eval()
    #_, predicted_val = torch.max(out[val_mask], dim=1)
    predicted_val = torch.argmax(out[val_mask], dim=1)
    precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(data.y[val_mask], predicted_val, average='macro')
    precisions.append(precision_val)
    recalls.append(recall_val)
    f1s.append(f1_val)
    # 计算负对数似然损失
    print("precision_val: %f, recall_val: %f, f1_val: %f, loss: %f" % (precision_val, recall_val, f1_val, loss.item()))
    #执行完model.eval()后从新开始train模式
    model.train()

# 训练过程参数变化可视化
plot_metrics(precisions, recalls, f1s, losses)




#预测部分
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




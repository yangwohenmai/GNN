import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import baostock as bs
import os
import sys
import random
import copy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_score, confusion_matrix, accuracy_score
import Strategy_BLJJ
from Strategy import TradeTag

sys.path.append('..')
from DataBase import StockPool
from DataBase import StockData
from DataBase import TrainData

lg = bs.login()
#stockPoolList = StockPool.GetStockPool('',False,'')
stockPriceDic = StockData.GetStockPriceDWMBaostock('000001.SZ', "20250901", 2000)
# 获取MACD数据 MainFuncBS:数据源baostock; MainFunc:数据源TS;
resultBLJJ = Strategy_BLJJ.GetBLJJFunc('000001.SZ', stockPriceDic, 2050, int(len(stockPriceDic)*0.9), "D", "close")["BLJJDic"]
#resultBLJJ = Strategy_BLJJ.MainFuncBS('000001.SZ', "20241101", 1450, len(stockPriceDic), "D", "close")["BLJJDic"]
#resultBLJJ = Strategy_BLJJ.MainFunc('000001.SZ', "20241101", 1450, len(stockPriceDic), "D", "close")["BLJJDic"]
if resultBLJJ == False:
    print("指标数据出错")
# 根据结果获取信号状态区间
buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'], resultBLJJ['longList'], resultBLJJ['shortList'])
# 匹配信号状态到每个交易日
newStockPriceDic = dict()
for key,f in stockPriceDic.items():
    date_obj = datetime.strptime(key, "%Y-%m-%d").strftime("%Y%m%d")
    if date_obj in buyAndSellPeriod['flagDic']:
        flag = buyAndSellPeriod['flagDic'][date_obj]
        if flag != -1:
            newStockPriceDic[key] = stockPriceDic[key]
            newStockPriceDic[key]['flag'] = flag
stockPriceDic = newStockPriceDic
# 构建图结构
data = TrainData.TrainDataMACD(stockPriceDic)[0]
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

# 特征标准化（仅用训练集统计量，防止测试集信息泄露）
x_data = data.x.to(torch.float32)
train_indices = torch.tensor([i for i in range(len(train_mask)) if train_mask[i]])
train_mean = x_data[train_indices].mean(dim=0, keepdim=True)
train_std = x_data[train_indices].std(dim=0, keepdim=True)
train_std[train_std == 0] = 1.0  # 防止除零
data.x = (x_data - train_mean) / train_std

# 计算训练集类别权重（缓解类别不平衡）
train_labels = data.y.to(torch.long)[train_indices]
class_counts = torch.bincount(train_labels, minlength=2)
class_weights = len(train_labels) / (2.0 * class_counts.float())
class_weights = class_weights / class_weights.sum()  # 归一化
print(f"训练集类别分布: 类0={class_counts[0].item()}, 类1={class_counts[1].item()}")
print(f"类别权重: 类0={class_weights[0].item():.4f}, 类1={class_weights[1].item():.4f}")

# 固定所有随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 升维阶段：7 -> 32 -> 64 -> 128 -> 128
        self.conv1 = GCNConv(5,   32,  add_self_loops=False)  # 第1层：感知前1天（输入5维特征）
        self.conv2 = GATConv(32,  64,  add_self_loops=False)  # 第2层：感知前2天
        self.conv3 = GCNConv(64,  128, add_self_loops=False)  # 第3层：感知前3天
        self.conv4 = GATConv(128, 128, add_self_loops=False)  # 第4层：感知前4天
        # 降维阶段：128 -> 64 -> 32 -> 16 -> 2
        self.conv5 = GCNConv(128, 64,  add_self_loops=False)  # 第5层：感知前5天
        self.conv6 = GATConv(64,  32,  add_self_loops=False)  # 第6层：感知前6天
        self.conv7 = GCNConv(32,  16,  add_self_loops=False)  # 第7层：感知前7天
        self.conv8 = GATConv(16,  2,   add_self_loops=False)  # 第8层：感知前8天，输出分类
        # 残差连接投影层：将输入维度对齐到输出维度，使残差可直接相加
        self.proj1 = torch.nn.Linear(5,   32,  bias=False)   # conv1残差：5->32
        self.proj3 = torch.nn.Linear(64,  128, bias=False)   # conv3残差：64->128（输入是conv2输出）
        self.proj5 = torch.nn.Linear(128, 64,  bias=False)   # conv4-5残差：128->64
        self.proj7 = torch.nn.Linear(32,  16,  bias=False)   # conv7残差：32->16（输入是conv6输出）
        # BatchNorm 稳定各阶段激活值，缓解梯度消失
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.bn6 = torch.nn.BatchNorm1d(32)
        self.bn7 = torch.nn.BatchNorm1d(16)

    def forward(self, x, edge_index):
        # 第1层：感知前1天 + 残差连接
        x = self.bn1(self.conv1(x, edge_index)) + self.proj1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第2层：感知前2天
        x = self.bn2(self.conv2(x, edge_index))
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第3层：感知前3天 + 残差连接（跳过conv2，从32直连到128）
        residual3 = self.proj3(x)
        x = self.bn3(self.conv3(x, edge_index)) + residual3
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第4层：感知前4天 + 残差连接（同维度直接相加）
        x = self.bn4(self.conv4(x, edge_index)) + x
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第5层：感知前5天 + 残差连接（128->64）
        residual5 = self.proj5(x)
        x = self.bn5(self.conv5(x, edge_index)) + residual5
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第6层：感知前6天
        x = self.bn6(self.conv6(x, edge_index))
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第7层：感知前7天 + 残差连接（跳过conv6，从64直连到16）
        residual7 = self.proj7(x)
        x = self.bn7(self.conv7(x, edge_index)) + residual7
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # 第8层：感知前8天，输出分类 logits
        x = self.conv8(x, edge_index)
        return F.log_softmax(x, dim=1)

set_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
# 学习率调度器：验证损失停滞时自动降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

# 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
precisions, recalls, f1s, losses = [], [], [], []
#模型训练/验证
model.train()
for epoch in range(800):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)    #特征已标准化，无需再转float32
    loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask], weight=class_weights.to(device))   #损失仅仅计算的是训练集的损失，加权
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    #启用验证模式，重新前向传播计算验证集指标
    model.eval()
    with torch.no_grad():
        out_val = model(data.x, data.edge_index)
        predicted_val = torch.argmax(out_val[val_mask], dim=1)
        val_loss = F.nll_loss(out_val[val_mask], data.y.to(torch.long)[val_mask], weight=class_weights.to(device)).item()
        precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(data.y.to(torch.long)[val_mask].cpu(), predicted_val.cpu(), average='macro', zero_division=0)
    precisions.append(precision_val)
    recalls.append(recall_val)
    f1s.append(f1_val)
    # 学习率调度
    scheduler.step(val_loss)
    # 计算负对数似然损失
    print("epoch: %d, precision_val: %f, recall_val: %f, f1_val: %f, loss: %f, val_loss: %f" % (epoch+1, precision_val, recall_val, f1_val, loss.item(), val_loss))
    #执行完model.eval()后从新开始train模式
    model.train()

# 训练过程参数变化可视化
#plot_metrics(precisions, recalls, f1s, losses)

# 登出baostock
bs.logout()

#预测部分
model.eval()
with torch.no_grad():
    test_predict = model(data.x, data.edge_index)[test_mask]
    max_index = torch.argmax(test_predict, dim=1)
    #test_true = data.y[data.test_mask]
    test_true = data.y.to(torch.long)[test_mask]
test_pred = max_index.cpu().numpy()
test_true_np = test_true.cpu().numpy()

# 计算多项评估指标
accuracy = accuracy_score(test_true_np, test_pred)
precision, recall, f1, _ = precision_recall_fscore_support(test_true_np, test_pred, average='macro', zero_division=0)
cm = confusion_matrix(test_true_np, test_pred)

print('==============================')
print('测试集评估结果')
print('==============================')
print('Accuracy:  {:.2f}%'.format(accuracy * 100))
print('Precision: {:.4f}'.format(precision))
print('Recall:    {:.4f}'.format(recall))
print('F1 (macro): {:.4f}'.format(f1))
print('------------------------------')
print('混淆矩阵 (行=真实, 列=预测):')
print(cm)
print('==============================')

# 训练过程参数变化可视化（按回车后显示图表）
input('\n按回车键查看训练指标曲线图...')
plot_metrics(precisions, recalls, f1s, losses)

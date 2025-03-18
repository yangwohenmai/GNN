import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import baostock as bs
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_score
import torch.nn as nn
import copy  # 用于保存最佳模型

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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(7, 32)
        self.conv2 = GATConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GATConv(128, 2)
        self.conv5 = GATConv(256, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
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

# Early stopping
patience = 800
best_model_wts = copy.deepcopy(model.state_dict())
best_precision = 0.0
counter = 0

precisions, recalls, f1s, losses = [], [], [], []
model.train()

for epoch in range(3500):
    optimizer.zero_grad()
    out = model(data.x.to(torch.float32), data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask])
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    # 启用验证模式
    model.eval()
    with torch.no_grad():
        val_loss = F.nll_loss(out[val_mask], data.y.to(torch.long)[val_mask]).item()
        predicted_val = torch.argmax(out[val_mask], dim=1)
        precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
            data.y[val_mask], predicted_val, average='macro')
        precisions.append(precision_val)
        recalls.append(recall_val)
        f1s.append(f1_val)

        # 保存最佳模型
        if precision_val > best_precision:
            best_precision = precision_val
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        # 打印训练信息
        print("Epoch: %d, val_loss: %f, precision_val: %f, recall_val: %f, f1_val: %f, train_loss: %f" % (epoch + 1, val_loss, precision_val, recall_val, f1_val, loss.item()))

        # 检查早停条件
        if counter >= patience:
            print("Early stopping at epoch %d" % (epoch + 1))
            break

    # 恢复训练模式
    model.train()

# 加载最佳模型参数
model.load_state_dict(best_model_wts)

# 训练过程参数变化可视化
plot_metrics(precisions, recalls, f1s, losses)

# 预测部分
test_predict = model(data.x.to(torch.float32), data.edge_index)[test_mask]
max_index = torch.argmax(test_predict, dim=1)
test_true = data.y.to(torch.long)[test_mask]
correct = 0
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
print('测试集准确率为：{}%'.format(correct*100/len(test_true)))

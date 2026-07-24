import torch
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
import baostock as bs
import os
import sys
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, precision_score, confusion_matrix, accuracy_score
import Strategy_BLJJ
from Strategy import TradeTag
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

#参数
dropoutRate = 0.3
trainingTimes = 2000 #训练轮次
printInterval = 30   #训练参数打印间隔
ifOpenNormalize = False #是否启用归一化（不开）
ifOpenEarlyStop = True    #是否启用早停（不开）
earlyStopPatience = 300   #连续多少轮验证F1未提升则停止
ifOpenLRScheduler = False  #是否启用学习率自动调整
lrPatience = 100           #验证F1多少轮未提升则降低学习率
lrFactor = 0.5             #每次降低到原来的比例
ifOpenEdgeDropout = False     #是否启用边Dropout
edgeDropoutRate = 0.2         #边Dropout丢弃率
ifOpenClassWeight = False    #是否启用类别加权损失
ifOpenBatchNorm = False      #是否启用BatchNorm


lg = bs.login()
#stockPoolList = StockPool.GetStockPool('',False,'')20250101
stockPriceDic = StockData.GetStockPriceDWMBaostock('000001.SZ', "20250101", 2000)
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
def normalize_features(data, train_mask):
    """
    对节点特征做标准化，消除量纲差异（仅 fit 训练集，防止数据泄露）
    :param data: PyG Data 对象
    :param train_mask: 训练集 mask（list[bool]）
    :return: data（原地修改后返回）
    """
    x_np = data.x.numpy().astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_np[train_mask, :5])            # 只用训练集 fit 前5列（open/close/low/high/pctChg）
    x_np[:, :5] = scaler.transform(x_np[:, :5]) # transform 全部数据
    data.x = torch.tensor(x_np, dtype=torch.float32)
    return data

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

# 记录和打印训练/验证进度
def log_training_progress(epoch, loss, model, data, train_mask, val_mask, trainingTimes, printInterval=50, best_f1=0):
    """
    计算训练/验证指标并格式化输出
    :return: precision_val, recall_val, f1_val, best_f1
    """
    model.eval()
    with torch.no_grad():
        out_val = model(data.x.to(torch.float32), data.edge_index)
        # 验证集指标
        predicted_val = torch.argmax(out_val[val_mask], dim=1)
        p_val, r_val, f1_val, _ = precision_recall_fscore_support(data.y.to(torch.long)[val_mask].cpu(), predicted_val.cpu(), average='macro')
        # 训练集指标
        predicted_tr = torch.argmax(out_val[train_mask], dim=1)
        _, _, f1_tr, _ = precision_recall_fscore_support(data.y.to(torch.long)[train_mask].cpu(), predicted_tr.cpu(), average='macro')

    is_best = ""
    if f1_val > best_f1:
        best_f1 = f1_val
        is_best = " *"

    if (epoch + 1) % printInterval == 0 or epoch == 0:
        print(f"[{epoch+1:4d}/{trainingTimes}] "
              f"loss={loss.item():.4f} | "
              f"train_F1={f1_tr:.4f} | "
              f"val_Precision={p_val:.4f} val_Recall={r_val:.4f} val_F1={f1_val:.4f}"
              f"{is_best}")

    return p_val, r_val, f1_val, best_f1

# 早停控制器
class EarlyStopper:
    """
    早停控制器：监控验证集F1，连续patience轮未提升则停止训练，并保存最佳模型权重
    """
    def __init__(self, patience=200):
        self.patience = patience
        self.best_f1 = 0.0
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, f1_val, model):
        """
        每轮训练后调用，返回是否应停止训练
        """
        if f1_val > self.best_f1:
            self.best_f1 = f1_val
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model):
        """
        训练结束后恢复最佳模型权重
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model

# 学习率调度器
def create_scheduler(optimizer, ifOpen, patience=100, factor=0.5):
    """
    创建学习率调度器，返回None表示不启用
    """
    if not ifOpen:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience)

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
        # add_self_loops=False：确保预测第N天时只使用前N-1天数据，节点i只聚合邻居i-1的特征
        self.conv1 = GCNConv(7, 32, add_self_loops=False)
        self.conv2 = GATConv(32, 64, add_self_loops=False)
        self.conv3 = GCNConv(64, 128, add_self_loops=False)
        self.conv4 = GATConv(128, 128, add_self_loops=False)
        self.conv5 = GCNConv(128, 128, add_self_loops=False)
        self.conv6 = GCNConv(128, 64, add_self_loops=False)
        self.conv7 = GATConv(64, 2, add_self_loops=False)
        self.dropout = torch.nn.Dropout(dropoutRate)
        self.edge_dropout_rate = edgeDropoutRate if ifOpenEdgeDropout else 0.0
        # 残差投影层：维度不匹配时做线性投影对齐
        self.proj1 = torch.nn.Linear(2, 32)  # 不含flag维，防止残差路径泄露标签
        self.proj2 = torch.nn.Linear(32, 64)
        self.proj3 = torch.nn.Linear(64, 128)
        self.proj6 = torch.nn.Linear(128, 64)
        # 是否启用BatchNorm
        if ifOpenBatchNorm:
            self.bn1 = torch.nn.BatchNorm1d(32)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.bn3 = torch.nn.BatchNorm1d(128)
            self.bn4 = torch.nn.BatchNorm1d(128)
            self.bn5 = torch.nn.BatchNorm1d(128)
            self.bn6 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x, edge_index):
        #训练时随机丢弃边，防止过度依赖特定邻居
        if self.training and self.edge_dropout_rate > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_rate)
        # conv1: 残差只用 open(dim0)+时间位置(dim5)，防止当天数据泄露
        res = self.proj1(x[:, [0, 5]])
        x = self.conv1(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn1(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv2: 维度不同，投影层对齐
        res = self.proj2(x)
        x = self.conv2(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn2(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv3: 维度不同，投影层对齐
        res = self.proj3(x)
        x = self.conv3(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn3(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv4: 维度相同直接相加
        res = x
        x = self.conv4(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn4(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv5: 维度相同直接相加
        res = x
        x = self.conv5(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn5(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv6: 维度不同，投影层对齐
        res = self.proj6(x)
        x = self.conv6(x, edge_index)
        if ifOpenBatchNorm:
            x = self.bn6(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv7: 输出层，不加残差
        x = self.conv7(x, edge_index)
        return F.log_softmax(x, dim=1)

set_seed(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
if ifOpenNormalize == True:
    data = normalize_features(data, train_mask) #数据归一化
data = data.to(device)
# 定义损失函数和优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
# 类别加权：用训练集统计各类别权重，平衡不平衡样本
if ifOpenClassWeight:
    from sklearn.utils.class_weight import compute_class_weight
    y_train = data.y.cpu().numpy()[train_mask]
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f'类别权重: {cw}')
else:
    class_weight_tensor = None
# 定义学习率调度器
scheduler = create_scheduler(optimizer, ifOpenLRScheduler, lrPatience, lrFactor)

# 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
precisions, recalls, f1s, losses = [], [], [], []
# 初始化早停控制器
early_stopper = EarlyStopper(earlyStopPatience) if ifOpenEarlyStop else None
# 最佳F1初始化，用于记录训练过程中最佳验证F1
best_f1 = 0.0
#模型训练/验证
model.train()
for epoch in range(trainingTimes):
    optimizer.zero_grad()
    out = model(data.x.to(torch.float32), data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
    loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask], weight=class_weight_tensor)   #损失仅仅计算的是训练集的损失
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    #启用验证模式，计算训练/验证指标并输出
    precision_val, recall_val, f1_val, best_f1 = log_training_progress(epoch, loss, model, data, train_mask, val_mask, trainingTimes, printInterval, best_f1)
    precisions.append(precision_val)
    recalls.append(recall_val)
    f1s.append(f1_val)
    #早停检测
    if early_stopper is not None:
        if early_stopper.step(f1_val, model):
            print(f"早停触发：连续{early_stopper.patience}轮验证F1未提升，停止训练于第{epoch+1}轮")
            break
    #学习率自动调整
    if scheduler is not None:
        scheduler.step(f1_val)
    #执行完model.eval()后从新开始train模式
    model.train()

#早停模式下恢复最佳模型权重
if early_stopper is not None:
    model = early_stopper.restore_best(model)
    print(f"已恢复最佳模型权重（验证F1={early_stopper.best_f1:.4f}）")

# 训练过程参数变化可视化
#plot_metrics(precisions, recalls, f1s, losses)

#预测部分
#test_predict = model(data.x.to(torch.float32), data.edge_index)[test_mask]
#max_index = torch.argmax(test_predict, dim=1)
#test_true = data.y.to(torch.long)[test_mask]
#correct = 0
#for i in range(len(max_index)):
#    if max_index[i] == test_true[i]:
#        correct += 1
#print('测试集准确率为：{}%'.format(correct*100/len(test_true)))

#预测部分
model.eval()
with torch.no_grad():
    #test_predict = model(data.x, data.edge_index)[data.test_mask]
    test_predict = model(data.x.to(torch.float32), data.edge_index)[test_mask]
    max_index = torch.argmax(test_predict, dim=1)
    #test_true = data.y[data.test_mask]
    test_true = data.y.to(torch.long)[test_mask]
test_pred = max_index.cpu().numpy()
test_true_np = test_true.cpu().numpy()

# 计算多项评估指标
accuracy = accuracy_score(test_true_np, test_pred)
precision, recall, f1, _ = precision_recall_fscore_support(test_true_np, test_pred, average='macro')
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
#input('\n按回车键查看训练指标曲线图...')
#plot_metrics(precisions, recalls, f1s, losses)
bs.logout()
import torch
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Batch
import baostock as bs
import os
import sys
import random
import time
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, precision_score, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
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
from Helper import LogHelper

#参数
# 最优参数 dropoutRate = 0.1 20250101 residualHistoryN = 5 1400 ifOpenNormalize = True
stockCode = '000001.SZ'
dataDate = "20250101"
maxStockCount = 30          # 最大处理股票数（None=不限制，建议先小规模验证再扩大）
dropoutRate = 0.1           # Dropout率
trainingTimes = 3000        # 训练轮次
printInterval = 30          # 训练参数打印间隔
ifOpenNormalize = True      # 是否启用归一化（不开）
ifOpenEarlyStop = True      # 是否启用早停（不开）
earlyStopPatience = 800     # 连续多少轮验证F1未提升则停止
ifOpenLRScheduler = False   # 是否启用学习率自动调整
lrPatience = 100            # 验证F1多少轮未提升则降低学习率
lrFactor = 0.5              # 每次降低到原来的比例
ifOpenEdgeDropout = False   # 是否启用边Dropout
edgeDropoutRate = 0.2       # 边Dropout丢弃率
ifOpenClassWeight = False   # 是否启用类别加权损失
ifOpenBatchNorm = False     # 是否启用BatchNorm
ifOpenFocalLoss = False     # 是否启用Focal Loss（动态聚焦难分样本，对抗类别塌缩）
focalLossGamma = 1.0        # Focal Loss聚焦参数（越大越聚焦难样本，通常取2）
residualHistoryN = 5        # 短残差历史窗口大小（1=仅x[i-1]，n=前n个历史节点x[i-n]~x[i-1]拼接后投影）
edgeWindowK = 1             # 入边窗口大小（每个节点i接收前K个相邻节点的边X[i-K]~X[i-1]→X[i]，1=单链结构）
edgeStride = 1              # 入边稀疏间隔（从X[i-1]开始每隔stride取一个，如K=3、stride=2时仅X[i-3]、X[i-1]→X[i]，1=稠密窗口）
ifOpenHyperSearch = True    # 是否启用超参数随机搜索（开启后搜索空间内参数的全局值失效，自动寻找最佳组合）
hyperSearchTrials = 10      # 随机搜索采样组数
hyperSearchTrainingTimes = 3000  #搜索阶段每组训练轮次（短轮次快速筛选，选出最佳组合后再用trainingTimes完整训练）
hyperSearchSpace = {        # 搜索空间：参数名→候选值列表（可自行增删候选值）
    'ifOpenNormalize':   [True],            #[True, False],
    'ifOpenClassWeight': [False],           #[True, False],
    'ifOpenBatchNorm':   [False],           #[True, False],
    'residualHistoryN':  [1, 3, 5, 8],
    'edgeWindowK':       [1, 5, 10, 20],
    'edgeStride':        [1, 2, 3, 5],
    'dropoutRate':       [0.1, 0.2, 0.4],
    'ifOpenEdgeDropout': [False],           #[True, False],
    'edgeDropoutRate':   [0.2],             #[0.1, 0.2, 0.3],
    'ifOpenFocalLoss':   [False],           #[True, False],
    'focalLossGamma':    [1.0],             #[1.0, 2.0],
    'earlyStopPatience': [200]              #[50, 100, 200]搜索阶段用小patience加速（单次训练模式用全局earlyStopPatience=800）
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #运行设备：有GPU用cuda，否则用cpu


lg = bs.login()

# 多股票预处理：每只股票独立处理（行情→BLJJ→flag→过滤→mask），收集后供run_training拼接成大图
def process_single_stock(code, endDate, period=1400):
    """
    处理单只股票：拉行情 → BLJJ → flag标注 → 过滤空窗 → mask构建
    :return: (priceDic, train_mask, val_mask, test_mask, code) 或 None（失败时）
    """
    stockPriceDic = StockData.GetStockPriceDWMBaostock(code, endDate, period)
    if stockPriceDic is False or len(stockPriceDic) < 50:
        return None
    resultBLJJ = Strategy_BLJJ.GetBLJJFunc(code, stockPriceDic, 1450, int(len(stockPriceDic)*0.9), "D", "close")["BLJJDic"]
    if resultBLJJ == False:
        return None
    buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'], resultBLJJ['longList'], resultBLJJ['shortList'])
    newStockPriceDic = dict()
    for key, f in stockPriceDic.items():
        date_obj = datetime.strptime(key, "%Y-%m-%d").strftime("%Y%m%d")
        if date_obj in buyAndSellPeriod['flagDic']:
            flag = buyAndSellPeriod['flagDic'][date_obj]
            if flag != -1:
                newStockPriceDic[key] = stockPriceDic[key]
                newStockPriceDic[key]['flag'] = flag
    if len(newStockPriceDic) < 50:
        return None
    split_train = int(len(newStockPriceDic) * 0.75)
    split_val = int(len(newStockPriceDic) * 0.85)
    train_mask, val_mask, test_mask = [], [], []
    for i in range(len(newStockPriceDic)):
        train_mask.append(i < split_train)
        val_mask.append(split_train <= i < split_val)
        test_mask.append(i >= split_val)
    return newStockPriceDic, train_mask, val_mask, test_mask, code

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
        acc_val = accuracy_score(data.y.to(torch.long)[val_mask].cpu(), predicted_val.cpu())
        # 训练集指标
        predicted_tr = torch.argmax(out_val[train_mask], dim=1)
        p_tr, r_tr, f1_tr, _ = precision_recall_fscore_support(data.y.to(torch.long)[train_mask].cpu(), predicted_tr.cpu(), average='macro')
        acc_tr = accuracy_score(data.y.to(torch.long)[train_mask].cpu(), predicted_tr.cpu())

    is_best = ""
    if f1_val > best_f1:
        best_f1 = f1_val
        is_best = " *"

    if printInterval > 0 and ((epoch + 1) % printInterval == 0 or epoch == 0):
        print(f"[{epoch+1:4d}/{trainingTimes}] loss={loss.item():.4f} | "
              f"train[Acc={acc_tr:.4f} P={p_tr:.4f} R={r_tr:.4f} F1={f1_tr:.4f}] | "
              f"val[Acc={acc_val:.4f} P={p_val:.4f} R={r_val:.4f} F1={f1_val:.4f}]{is_best}")

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

# Focal Loss：动态聚焦难分样本，比类别加权更强地对抗类别塌缩
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重张量，shape [num_classes]
        self.gamma = gamma  # 聚焦参数

    def forward(self, log_probs, targets):
        probs = torch.exp(log_probs)
        targets = targets.long()
        p_t = probs[range(len(targets)), targets]
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_weight * log_probs[range(len(targets)), targets]
        else:
            loss = -focal_weight * log_probs[range(len(targets)), targets]
        return loss.mean()

#print(data.x)
#print(data.y)
#print(data.x[0].tolist())
#print(data.y.tolist())
#print(data.edge_index[0].tolist())
#print(data.edge_index[1].tolist())
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: 超参数字典（dropoutRate/ifOpenBatchNorm/residualHistoryN/ifOpenEdgeDropout/edgeDropoutRate等）
        """
        super(Net, self).__init__()
        # add_self_loops=False：确保预测第N天时只使用前N-1天数据，节点i只聚合邻居i-1的特征
        # 10层网络：5个Block，每2层一个Block，维度平滑过渡 7→32→32→64→64→128→128→128→128→64→2
        self.conv1 = GCNConv(7, 32, add_self_loops=False)
        self.conv2 = GATConv(32, 32, add_self_loops=False)
        self.conv3 = GCNConv(32, 64, add_self_loops=False)
        self.conv4 = GATConv(64, 64, add_self_loops=False)
        self.conv5 = GCNConv(64, 128, add_self_loops=False)
        self.conv6 = GATConv(128, 128, add_self_loops=False)
        self.conv7 = GCNConv(128, 128, add_self_loops=False)
        self.conv8 = GATConv(128, 128, add_self_loops=False)
        self.conv9 = GCNConv(128, 64, add_self_loops=False)
        self.conv10 = GATConv(64, 2, add_self_loops=False)
        self.dropout = torch.nn.Dropout(cfg['dropoutRate'])
        self.edge_dropout_rate = cfg['edgeDropoutRate'] if cfg['ifOpenEdgeDropout'] else 0.0
        self.residualHistoryN = cfg['residualHistoryN']
        # 短残差投影层（维度不匹配时做线性投影对齐）
        # residualHistoryN=1时输入7维；n>1时拼接n个历史节点特征，输入7*n维
        self.proj1 = torch.nn.Linear(7 * cfg['residualHistoryN'], 32)    # conv1残差（前n个历史节点特征拼接后投影）
        self.proj3 = torch.nn.Linear(32, 64)   # conv3残差
        self.proj5 = torch.nn.Linear(64, 128)  # conv5残差
        self.proj9 = torch.nn.Linear(128, 64)  # conv9残差
        # 是否启用BatchNorm
        self.ifOpenBatchNorm = cfg['ifOpenBatchNorm']
        if self.ifOpenBatchNorm:
            self.bn1 = torch.nn.BatchNorm1d(32)
            self.bn2 = torch.nn.BatchNorm1d(32)
            self.bn3 = torch.nn.BatchNorm1d(64)
            self.bn4 = torch.nn.BatchNorm1d(64)
            self.bn5 = torch.nn.BatchNorm1d(128)
            self.bn6 = torch.nn.BatchNorm1d(128)
            self.bn7 = torch.nn.BatchNorm1d(128)
            self.bn8 = torch.nn.BatchNorm1d(128)
            self.bn9 = torch.nn.BatchNorm1d(64)

    def forward(self, x, edge_index):
        #训练时随机丢弃边，防止过度依赖特定邻居
        if self.training and self.edge_dropout_rate > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_rate)
        # === Block 1: conv1 + conv2（32维平台，含跨层残差） ===
        # conv1: 短残差使用前residualHistoryN个历史节点的特征拼接（shift排除当日x[i]），防止数据泄露
        # n=1时: shifted_x[i] = x[i-1]（当前行为）
        # n>1时: 拼接 x[i-n], x[i-n+1], ..., x[i-1]（缺失位置补零向量）
        shifted_list = []
        for k in range(self.residualHistoryN, 0, -1):
            shifted_k = torch.zeros_like(x)
            shifted_k[k:] = x[:-k]
            shifted_list.append(shifted_k)
        shifted_x = torch.cat(shifted_list, dim=1)
        res = self.proj1(shifted_x)
        x = self.conv1(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn1(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip1 = x  # conv1输出(32维)，供Block1跨层残差使用
        # conv2: 短残差(32→32直接相加) + 跨层残差(conv1输出→conv2输出, 32→32直接相加)
        res = x
        x = self.conv2(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn2(x)
        x = F.relu(x + res + skip1)
        x = self.dropout(x)
        # === Block 2: conv3 + conv4（64维平台，含跨层残差） ===
        # conv3: 短残差
        res = self.proj3(x)
        x = self.conv3(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn3(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip3 = x  # conv3输出(64维)，供Block2跨层残差使用
        # conv4: 短残差(64→64直接相加) + 跨层残差(conv3输出→conv4输出, 64→64直接相加)
        res = x
        x = self.conv4(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn4(x)
        x = F.relu(x + res + skip3)
        x = self.dropout(x)
        # === Block 3: conv5 + conv6（128维平台，含跨层残差） ===
        # conv5: 短残差
        res = self.proj5(x)
        x = self.conv5(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn5(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip5 = x  # conv5输出(128维)，供Block3跨层残差使用
        # conv6: 短残差(128→128直接相加) + 跨层残差(conv5输出→conv6输出, 128→128直接相加)
        res = x
        x = self.conv6(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn6(x)
        x = F.relu(x + res + skip5)
        x = self.dropout(x)
        # === Block 4: conv7 + conv8（128维平台，含跨层残差） ===
        # conv7: 短残差(128→128直接相加)
        res = x
        x = self.conv7(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn7(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip7 = x  # conv7输出(128维)，供Block4跨层残差使用
        # conv8: 短残差 + 跨层残差(conv7输出→conv8输出, 128→128直接相加)
        res = x
        x = self.conv8(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn8(x)
        x = F.relu(x + res + skip7)
        x = self.dropout(x)
        # === Block 5: conv9 + conv10（降维+输出） ===
        # conv9: 短残差
        res = self.proj9(x)
        x = self.conv9(x, edge_index)
        if self.ifOpenBatchNorm: x = self.bn9(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv10: 输出层，不加残差
        x = self.conv10(x, edge_index)
        return F.log_softmax(x, dim=1)

# 按超参数配置执行一次完整流程：建图→归一化→建模→训练(早停)→测试评估
def run_training(cfg, stock_data_list, quiet=False, epochs=None):
    """
    :param cfg: 超参数字典（含搜索空间的10个参数）
    :param stock_data_list: 多股票预处理结果列表，每个元素为(priceDic, train_mask, val_mask, test_mask, code)
    :param quiet: True时静默运行（超参数搜索阶段用，不打印逐轮日志）
    :param epochs: 训练轮次（None时使用全局trainingTimes）
    :return: dict(best_val_f1/accuracy/precision/recall/f1/cm/model/训练过程指标列表)
    """
    set_seed(2)  # 每组配置从相同随机状态出发，保证公平对比
    epochs = trainingTimes if epochs is None else epochs
    # 多股票建图：每只股票按cfg的K/stride独立建图，再拼成一张大图（跨股票无边相连，信息不跨股票流动）
    data_list = []
    train_mask, val_mask, test_mask = [], [], []
    for priceDic, tr_mask, va_mask, te_mask, code in stock_data_list:
        d = TrainData.TrainDataMACDWindowK(priceDic, cfg['edgeWindowK'], cfg['edgeStride'])[0]
        data_list.append(d)
        train_mask.extend(tr_mask)
        val_mask.extend(va_mask)
        test_mask.extend(te_mask)
    data = Batch.from_data_list(data_list)
    model = Net(cfg).to(device)
    if cfg['ifOpenNormalize'] == True:
        data = normalize_features(data, train_mask) #数据归一化
    data = data.to(device)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    # 类别加权：用训练集统计各类别权重，平衡不平衡样本
    if cfg['ifOpenClassWeight']:
        from sklearn.utils.class_weight import compute_class_weight
        y_train = data.y.cpu().numpy()[train_mask]
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
        if not quiet:
            print(f'类别权重: {cw}')
    else:
        class_weight_tensor = None
    # 定义学习率调度器
    scheduler = create_scheduler(optimizer, ifOpenLRScheduler, lrPatience, lrFactor)
    focal_loss_fn = FocalLoss(alpha=class_weight_tensor, gamma=cfg['focalLossGamma']) if cfg['ifOpenFocalLoss'] else None
    if not quiet:
        print(f'本次训练配置: {cfg}')
        print(f'全局配置: 股票数={len(stock_data_list)}, 训练轮次={epochs}, 早停={ifOpenEarlyStop}(patience={cfg.get("earlyStopPatience", earlyStopPatience)}), 学习率调度={ifOpenLRScheduler}')
        if cfg['ifOpenFocalLoss']:
            print(f"Focal Loss已启用: gamma={cfg['focalLossGamma']}, alpha={class_weight_tensor}")
        if cfg['residualHistoryN'] > 1:
            print(f"短残差历史窗口: {cfg['residualHistoryN']}步拼接（维度 {7*cfg['residualHistoryN']}→32）")
        print(f"入边窗口: K={cfg['edgeWindowK']}, 稀疏间隔={cfg['edgeStride']}（每节点直接聚合前{cfg['edgeWindowK']}天内隔{cfg['edgeStride']}取一，边数={data.edge_index.shape[1]}）")

    # 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
    precisions, recalls, f1s, losses = [], [], [], []
    # 初始化早停控制器
    early_stopper = EarlyStopper(cfg.get('earlyStopPatience', earlyStopPatience)) if ifOpenEarlyStop else None
    # 最佳F1初始化，用于记录训练过程中最佳验证F1及其出现轮次
    best_f1 = 0.0
    best_epoch = 0
    #模型训练/验证
    train_start = time.time()   #记录训练开始时间，用于统计耗时
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x.to(torch.float32), data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
        #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])   #损失仅仅计算的是训练集的损失
        if focal_loss_fn is not None:
            loss = focal_loss_fn(out[train_mask], data.y.to(torch.long)[train_mask])
        else:
            loss = F.nll_loss(out[train_mask], data.y.to(torch.long)[train_mask], weight=class_weight_tensor)   #损失仅仅计算的是训练集的损失
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        #启用验证模式，计算训练/验证指标并输出（quiet时printInterval=0不打印）
        prev_best_f1 = best_f1
        precision_val, recall_val, f1_val, best_f1 = log_training_progress(epoch, loss, model, data, train_mask, val_mask, epochs, 0 if quiet else printInterval, best_f1)
        if best_f1 > prev_best_f1:
            best_epoch = epoch + 1  #记录最佳验证F1出现的轮次
        precisions.append(precision_val)
        recalls.append(recall_val)
        f1s.append(f1_val)
        #早停检测
        if early_stopper is not None:
            if early_stopper.step(f1_val, model):
                if not quiet:
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
        if not quiet:
            print(f"已恢复最佳模型权重（验证F1={early_stopper.best_f1:.4f}）")

    #训练耗时统计
    train_elapsed = time.time() - train_start
    if not quiet:
        print(f"训练完成：最佳验证F1={best_f1:.4f}（第{best_epoch}轮），耗时 {int(train_elapsed//60)}分{train_elapsed%60:.0f}秒")

    #测试集评估
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
    return {'best_val_f1': best_f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'cm': cm, 'model': model, 'best_epoch': best_epoch, 'elapsed': train_elapsed,
            'precisions': precisions, 'recalls': recalls, 'f1s': f1s, 'losses': losses}

# 边参数组合约束：要求edgeStride*2 < edgeWindowK（保证窗口内至少3条入边，避免大量组合退化成单链）
# 例外：edgeWindowK=1且edgeStride=1的单链基准组合放行
def valid_edge_combo(cfg):
    k = cfg.get('edgeWindowK', edgeWindowK)
    s = cfg.get('edgeStride', edgeStride)
    if k == 1 and s == 1:
        return True
    return s * 2 < k

# 从搜索空间随机采样不重复的超参数组合
def sample_configs(space, nTrials):
    # 已经抽过的参数组合存起来
    configs = []
    seen = set()
    attempts = 0
    # 随机采样不重复的超参数组合，抽了nTrials * 100次参数组合发现都是重复的，就不在抽了，说明基本都抽完了
    while len(configs) < nTrials and attempts < nTrials * 100:
        attempts += 1
        cfg = {k: random.choice(v) for k, v in space.items()}
        if not valid_edge_combo(cfg):
            continue
        key = tuple(str(cfg[k]) for k in sorted(cfg))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs

# 遍历码表获取所有股票（每只股票内部按时序75/10/15划分train/val/test，后续拼成大图时各mask拼接）
date = datetime.fromordinal(datetime.today().toordinal() - (datetime.today().weekday() or 7)).strftime('%Y-%m-%d')
stockPoolList = StockPool.GetHS300StockListBaostock()
stock_data_list = []  # 每个元素: (priceDic, train_mask, val_mask, test_mask, code)

dataCount = 0
for code in StockPool.GetALLStockListBaostock(date).keys():
    if len(stockPoolList) == 0 or code not in stockPoolList:
        continue
    if maxStockCount is not None and dataCount >= maxStockCount:
        break
    try:
        result = process_single_stock(code, dataDate, 1400)
        if result is not None:
            stock_data_list.append(result)
            dataCount += 1
            print(f'{code} 预处理完成,序号:NO.{dataCount},节点数:{len(result[0])}')
        else:
            print(code + ' 数据不足或指标出错,跳过')
    except Exception as ex:
        print("失败代码："+code+"，异常信息："+str(ex))
print(f'共预处理 {len(stock_data_list)} 只股票，总节点数 {sum(len(r[0]) for r in stock_data_list)}')

#主流程：超参数搜索模式 / 单次训练模式
if ifOpenHyperSearch:
    # 搜索模式下屏蔽sklearn的UndefinedMetricWarning（某类无预测样本时的警告），避免刷屏干扰每组摘要行；单次训练模式不屏蔽
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    # 固定采样种子，保证搜索组合可复现（需在run_training重置种子前一次性采样完，采样与训练互不影响）
    set_seed(2)
    trial_configs = sample_configs(hyperSearchSpace, hyperSearchTrials)
    # 强制包含单链基准组合（edgeWindowK=1且edgeStride=1，其余参数沿用第1组），作为窗口结构的对照基准
    baseline_cfg = dict(trial_configs[0])
    baseline_cfg['edgeWindowK'] = 1
    baseline_cfg['edgeStride'] = 1
    trial_configs[0] = baseline_cfg
    print(f'========== 超参数随机搜索：共{len(trial_configs)}组，每组{hyperSearchTrainingTimes}轮 ==========')
    trial_results = []
    for idx, cfg in enumerate(trial_configs):
        r = run_training(cfg, stock_data_list, quiet=True, epochs=hyperSearchTrainingTimes)
        trial_results.append((r['best_val_f1'], r, cfg))
        print(f"[trial {idx+1:2d}/{len(trial_configs)}] valF1={r['best_val_f1']:.4f}(第{r['best_epoch']}轮) | test[Acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}] 耗时={r['elapsed']:.0f}s  {cfg}")
    #按验证F1排序选最优（不看testF1，避免用测试集选模型造成评估泄露）
    trial_results.sort(key=lambda t: t[0], reverse=True)
    print('------ 搜索结果Top5（按验证F1排序） ------')
    for vf1, r, cfg in trial_results[:5]:
        print(f"valF1={vf1:.4f} testF1={r['f1']:.4f}  {cfg}")
    best_cfg = trial_results[0][2]
    print(f'最佳配置: {best_cfg}')
    print('提示：将最佳配置手动填回参数区并关闭ifOpenHyperSearch，即可单次训练复现（种子固定，结果与搜索时一致，可看逐轮日志与训练曲线）')
    result = trial_results[0][1]  #直接使用搜索中最佳组的结果，不再重复精训
else:
    #单次训练模式：使用参数区的全局配置（与原有行为一致）
    base_cfg = {
        'ifOpenNormalize': ifOpenNormalize,
        'ifOpenClassWeight': ifOpenClassWeight,
        'ifOpenBatchNorm': ifOpenBatchNorm,
        'residualHistoryN': residualHistoryN,
        'edgeWindowK': edgeWindowK,
        'edgeStride': edgeStride,
        'dropoutRate': dropoutRate,
        'ifOpenEdgeDropout': ifOpenEdgeDropout,
        'edgeDropoutRate': edgeDropoutRate,
        'ifOpenFocalLoss': ifOpenFocalLoss,
        'focalLossGamma': focalLossGamma,
        'earlyStopPatience': earlyStopPatience,
    }
    result = run_training(base_cfg, stock_data_list, quiet=False)

# 训练过程参数变化可视化
#plot_metrics(result['precisions'], result['recalls'], result['f1s'], result['losses'])

print('==============================')
print('测试集评估结果')
print('==============================')
print('Accuracy:  {:.2f}%'.format(result['accuracy'] * 100))
print('Precision: {:.4f}'.format(result['precision']))
print('Recall:    {:.4f}'.format(result['recall']))
print('F1 (macro): {:.4f}'.format(result['f1']))
print('------------------------------')
print('混淆矩阵 (行=真实, 列=预测):')
print(result['cm'])
print('==============================')

# 训练过程参数变化可视化（按回车后显示图表）
#input('\n按回车键查看训练指标曲线图...')
#plot_metrics(result['precisions'], result['recalls'], result['f1s'], result['losses'])
bs.logout()

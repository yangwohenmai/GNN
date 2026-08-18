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
stockCode = '000046.SZ'
dataDate = "20250101"       # 训练数据取值范围的截止日期
periodRange = 1400          # 根据dataDate，向前取多少个自然日
# 获取最新日期，取出当天所有股票作为股票池（默认取周一的股票池）
getNewStockPoolByDate = datetime.fromordinal(datetime.today().toordinal() - (datetime.today().weekday() or 7)).strftime('%Y-%m-%d')
ifOpenMultiStock = False    # 是否启用多股票训练（True=遍历沪深300码表拼大图，False=仅用stockCode单股票训练）
maxStockCount = 100          # 用多少只股票同时训练（仅多股票模式生效，None=不限制）
dropoutRate = 0.1           # Dropout率
trainingTimes = 6000        # 训练轮次
printInterval = 30          # 训练参数打印间隔
ifOpenNormalize = True      # 是否启用归一化（不开）
ifOpenEarlyStop = True      # 是否启用早停（不开）
earlyStopPatience = 800     # 连续多少轮验证F1未提升则停止
ifOpenLRScheduler = False   # 是否启用学习率自动调整
lrPatience = 100            # 验证F1多少轮未提升则降低学习率
lrFactor = 0.5              # 每次降低到原来的比例
lrMinLr = 1e-5              # 学习率下限（降到此值后不再降低，防止lr过小模型停止学习）
ifOpenEdgeDropout = False   # 是否启用边Dropout
edgeDropoutRate = 0.2       # 边Dropout丢弃率
ifOpenClassWeight = False   # 是否启用类别加权损失
ifOpenBatchNorm = False     # 是否启用BatchNorm
ifOpenFocalLoss = False     # 是否启用Focal Loss（动态聚焦难分样本，对抗类别塌缩）
focalLossGamma = 1.0        # Focal Loss聚焦参数（越大越聚焦难样本，通常取2）
residualHistoryN = 5        # 短残差历史窗口大小（1=仅x[i-1]，n=前n个历史节点x[i-n]~x[i-1]拼接后投影）
edgeWindowK =21            # 入边窗口大小（每个节点i接收前K个相邻节点的边X[i-K]~X[i-1]→X[i]，1=单链结构）
edgeStride = 3              # 入边稀疏间隔（从X[i-1]开始每隔stride取一个，如K=3、stride=2时仅X[i-3]、X[i-1]→X[i]，1=稠密窗口）
ifOpenAttentionHeatmap = True  # 是否在训练结束后绘制GAT层热力图（需edgeWindowK>1才有意义，K=1时每节点仅1条入边注意力恒为1）
netMode = 'mixed'           # 网络结构模式：mixed(GCN-GAT交替，当前默认)/onlyGCN(全GCNConv)/onlyGAT(全GATConv)
ifOpenAblation = True       # 是否启用消融实验模式（开启后遍历ablationModes各组训练并输出对比表，量化GCN/GAT对训练的影响）
ablationModes = ['mixed', 'onlyGCN', 'onlyGAT']  # 消融实验对比的网络模式列表（mixed=当前GCN-GAT交替基准）
ifOpenHyperSearch = False    # 是否启用超参数随机搜索（开启后搜索空间内参数的全局值失效，自动寻找最佳组合）
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


# 股票预处理：每只股票独立处理（行情→BLJJ→flag→过滤→mask），收集后供run_training拼接成大图
def process_single_stock(code, endDate, period=1400):
    """
    处理单只股票：拉行情 → BLJJ → flag标注 → 过滤空窗 → mask构建
    :return: (priceDic, train_mask, val_mask, test_mask, code) 或 None（失败时）
    """
    stockPriceDic = StockData.GetStockPriceDWMBaostock(code, endDate, period)
    if stockPriceDic is False or len(stockPriceDic) < 50:
        return None
    resultBLJJ = Strategy_BLJJ.GetBLJJFunc(code, stockPriceDic, period+50, int(len(stockPriceDic)*0.9), "D", "close")["BLJJDic"]
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

# GAT层热力图：滞后lag×时间，展示"预测第i天时对前K天历史的注意力分配"
# 因add_self_loops=False，每节点入边注意力经softmax后和为1，各列颜色分布可直接横向对比
def plot_attention_heatmaps(model, data, priceDic, cfg, train_mask, val_mask, stock_code='', mode=''):
    """
    绘制两张图：1) 每个GAT层一张“滞后×时间”热力图；2) 收盘价/标签与5层平均注意力的对齐视图
    :param model: 训练好的模型（已恢复最佳权重）
    :param data: 单只股票的图数据（多股票大图会导致股票交界处滞后错乱，须传单股票图）
    :param priceDic: 该股票行情字典（键=日期，顺序与节点一致），用于价格曲线与横轴日期
    :param cfg: 本次训练使用的超参数字典（取edgeWindowK）
    :param train_mask/val_mask: 该股票的划分mask，用于画 train/val/test 分界线
    """
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # no longer needed for English
    # plt.rcParams['axes.unicode_minus'] = False
    K = cfg['edgeWindowK']
    N = data.x.shape[0]
    # 所有层名（conv1~conv10）：GAT层用注意力权重，GCN层用伪注意力（特征相似度），全部显示
    layer_names = [f'conv{i+1}' for i in range(10)]
    # 用hook获取每层实际输入特征（经残差/dropout等处理后的真实输入，比手动逐层前向更准确）
    layer_inputs = {}
    def _pre_hook(name):
        def hook(module, inp):
            layer_inputs[name] = inp[0].clone()  # inp[0]=x（GCNConv/GATConv的输入）
        return hook
    handles = [getattr(model, f'conv{i+1}').register_forward_pre_hook(_pre_hook(f'conv{i+1}')) for i in range(10)]
    # 前向一次收集GAT层注意力权重 + 每层输入特征
    model.eval()
    with torch.no_grad():
        _, att_list = model(data.x, data.edge_index, return_attention=True)
    for h in handles:
        h.remove()
    # 边信息（用于GCN伪注意力计算）
    edge_index_np = data.edge_index.cpu().numpy()
    src, dst = edge_index_np[0], edge_index_np[1]
    # 对每层构造 [K, N] 矩阵：GAT层用注意力α，GCN层用特征相似度伪注意力
    layer_mats = []
    gat_idx = 0
    for i in range(10):
        name = f'conv{i+1}'
        mat = np.full((K, N), np.nan)
        if model.is_gat[i]:
            # GAT层：用学到的注意力权重
            ei, alpha = att_list[gat_idx]
            gat_idx += 1
            ei = ei.cpu().numpy()
            a = alpha.cpu().numpy().mean(axis=1)
            lag = ei[1] - ei[0]
            valid = (lag >= 1) & (lag <= K)
            mat[lag[valid] - 1, ei[1][valid]] = a[valid]
        else:
            # GCN层：基于输入特征的余弦相似度计算伪注意力
            feat = layer_inputs[name].cpu().numpy()
            norm = np.linalg.norm(feat, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-8)
            feat_norm = feat / norm
            sim = np.sum(feat_norm[src] * feat_norm[dst], axis=1)
            for node_i in range(N):
                mask = (dst == node_i)
                if not np.any(mask):
                    continue
                neighbor_indices = np.where(mask)[0]
                neighbor_nodes = src[neighbor_indices]
                neighbor_sims = sim[neighbor_indices]
                neighbor_sims = neighbor_sims - np.max(neighbor_sims)
                exp_sims = np.exp(neighbor_sims)
                sum_exp = np.sum(exp_sims)
                if sum_exp < 1e-8:
                    continue
                normalized = exp_sims / sum_exp
                for idx, node_j in enumerate(neighbor_nodes):
                    lag = node_i - node_j
                    if 1 <= lag <= K:
                        mat[lag - 1, node_i] = normalized[idx]
        layer_mats.append(mat)
    # 裁剪前K-1列：以最远lag=K的起点为准，对齐所有lag行的起点，避免左侧参差NaN
    crop_start = K - 1
    split_train = max(0, sum(train_mask) - crop_start)
    # 只取训练集部分
    layer_mats = [mat[:, crop_start:crop_start + split_train] for mat in layer_mats]
    N_cropped = split_train
    dates = list(priceDic.keys())[crop_start:crop_start + split_train]
    tick_pos = list(range(0, N_cropped, max(1, N_cropped // 8)))
    lag_ticks = list(range(0, K, max(1, K // 5)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')     # 无边位置（序列开头不足K天）显示为灰色

    # 图1：每个GAT层一张 滞后×时间 热力图（纵向对比各层注意力分布随深度的变化）
    fig1, axes = plt.subplots(len(layer_mats), 1, figsize=(14, 2.2 * len(layer_mats)), sharex=True, constrained_layout=True)
    for ax, mat, name in zip(axes, layer_mats, layer_names):
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_ylabel(f'{name}\nday')
        ax.set_yticks(lag_ticks)
        ax.set_yticklabels([str(l + 1) for l in lag_ticks])
        fig1.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels([dates[p] for p in tick_pos], rotation=30)
    axes[0].set_title(f'{stock_code} Layer Heatmaps - Training Data (GAT=attention weight, GCN=pseudo-attention)')

    # 图2：收盘价/标签(上) + 多层平均热力图(下)，对齐时间轴观察注意力突变与行情/信号的关系
    mean_mat = np.nanmean(np.stack(layer_mats), axis=0)
    closes = [float(v['close']) for v in priceDic.values()][crop_start:crop_start + split_train]
    y_np = data.y.cpu().numpy()[crop_start:crop_start + split_train]
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    ax1.plot(range(N_cropped), closes, color='gray', linewidth=0.8, label='close')
    idx1 = np.where(y_np == 1)[0]
    idx0 = np.where(y_np == 0)[0]
    ax1.scatter(idx1, [closes[i] for i in idx1], s=4, c='red', label='Label 1')
    ax1.scatter(idx0, [closes[i] for i in idx0], s=4, c='green', label='Label 0')
    ax1.set_ylabel('Close')
    ax1.set_title(f'{stock_code} Close/Label (top) vs Multi-layer Avg Attention (bottom) - Training Data')
    im2 = ax2.imshow(mean_mat, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest')
    ax2.set_ylabel('day')
    ax2.set_yticks(lag_ticks)
    ax2.set_yticklabels([str(l + 1) for l in lag_ticks])
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([dates[p] for p in tick_pos], rotation=30)
    fig2.colorbar(im2, ax=[ax1, ax2], pad=0.01)

    # 先保存再显示（防止关图后丢失，训练成本高），文件名以时间戳为前缀，方便多次训练区分
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    f1_name = f'{ts}_{mode}_GAT_heatmap_layers_{stock_code}.png'
    f2_name = f'{ts}_{mode}_GAT_heatmap_price_aligned_{stock_code}.png'
    fig1.savefig(f1_name, dpi=150)
    fig2.savefig(f2_name, dpi=150)
    print(f'Heatmaps saved: {f1_name} / {f2_name}')
    plt.show()

# GCN伪注意力热力图：基于特征相似度计算每个节点对邻居的关注度
# 格式与GAT层热力图一致（lag×时间），方便直接对比GAT和GCN的邻居关注度差异
def plot_gcn_implicit_attention_heatmap(model, data, priceDic, cfg, train_mask, val_mask, stock_code='', mode=''):
    """
    绘制GCN伪注意力热力图：基于输入特征的余弦相似度计算每个节点对邻居的关注度
    :param model: 训练好的模型
    :param data: 单只股票的图数据
    :param priceDic: 该股票行情字典
    :param cfg: 超参数字典（取edgeWindowK）
    :param train_mask/val_mask: 该股票的划分mask
    """
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # no longer needed for English
    # plt.rcParams['axes.unicode_minus'] = False
    K = cfg['edgeWindowK']
    N = data.x.shape[0]
    # 获取GCN层名（非GAT层）
    gcn_layer_names = [f'conv{i+1}' for i, g in enumerate(model.is_gat) if not g]
    if not gcn_layer_names:
        print('GCN pseudo-attention heatmap skipped: no GCN layers in current network')
        return
    
    # 提取GCN层对象和边信息
    gcn_layers = [getattr(model, name) for name in gcn_layer_names]
    edge_index = data.edge_index.cpu().numpy()
    src, dst = edge_index[0], edge_index[1]
    
    # 对每个GCN层，计算伪注意力
    layer_mats = []
    model.eval()
    with torch.no_grad():
        # 逐层前向，获取每层的输入特征
        x = data.x.clone()
        for layer_idx, (layer, name) in enumerate(zip(gcn_layers, gcn_layer_names)):
            # 当前层的输入特征
            x_in = x.clone()
            # 前向一层获取输出（用于下一层输入）
            x = layer(x, data.edge_index)
            # 应用ReLU和dropout（与forward一致）
            if layer_idx < len(gcn_layers) - 1:  # 最后一层不加ReLU
                x = F.relu(x)
            
            # 计算伪注意力：基于输入特征的余弦相似度
            feat = x_in.cpu().numpy()  # [N, D]
            # 归一化特征
            norm = np.linalg.norm(feat, axis=1, keepdims=True)
            norm = np.maximum(norm, 1e-8)
            feat_norm = feat / norm  # [N, D]
            
            # 对所有边计算余弦相似度（向量化）
            sim = np.sum(feat_norm[src] * feat_norm[dst], axis=1)  # [E]
            
            # 对每个节点，将其邻居的相似度归一化（softmax）
            mat = np.full((K, N), np.nan)
            for node_i in range(N):
                # 获取指向node_i的边索引
                mask = (dst == node_i)
                if not np.any(mask):
                    continue
                neighbor_indices = np.where(mask)[0]
                neighbor_nodes = src[neighbor_indices]
                neighbor_sims = sim[neighbor_indices]
                
                # softmax归一化
                neighbor_sims = neighbor_sims - np.max(neighbor_sims)  # 数值稳定
                exp_sims = np.exp(neighbor_sims)
                sum_exp = np.sum(exp_sims)
                if sum_exp < 1e-8:
                    continue
                normalized = exp_sims / sum_exp
                
                # 按lag组织到矩阵
                for idx, node_j in enumerate(neighbor_nodes):
                    lag = node_i - node_j
                    if 1 <= lag <= K:
                        mat[lag - 1, node_i] = normalized[idx]
            
            layer_mats.append(mat)
    
    # 裁剪前K-1列并只取训练集部分
    crop_start = K - 1
    split_train = max(0, sum(train_mask) - crop_start)
    layer_mats = [mat[:, crop_start:crop_start + split_train] for mat in layer_mats]
    N_cropped = split_train
    dates = list(priceDic.keys())[crop_start:crop_start + split_train]
    tick_pos = list(range(0, N_cropped, max(1, N_cropped // 8)))
    lag_ticks = list(range(0, K, max(1, K // 5)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')
    
    # 图1：每个GCN层一张 滞后×时间 热力图
    fig1, axes = plt.subplots(len(layer_mats), 1, figsize=(14, 2.2 * len(layer_mats)), sharex=True, constrained_layout=True)
    for ax, mat, name in zip(axes, layer_mats, gcn_layer_names):
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest')
        ax.set_ylabel(f'{name}\nday')
        ax.set_yticks(lag_ticks)
        ax.set_yticklabels([str(l + 1) for l in lag_ticks])
        fig1.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels([dates[p] for p in tick_pos], rotation=30)
    axes[0].set_title(f'{stock_code} GCN Pseudo-attention Heatmaps - Training Data (y-axis=lookback days, color=normalized similarity weight)')
    
    # 图2：收盘价/标签(上) + 多层平均伪注意力热力图(下)
    mean_mat = np.nanmean(np.stack(layer_mats), axis=0)
    closes = [float(v['close']) for v in priceDic.values()][crop_start:crop_start + split_train]
    y_np = data.y.cpu().numpy()[crop_start:crop_start + split_train]
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    ax1.plot(range(N_cropped), closes, color='gray', linewidth=0.8, label='close')
    idx1 = np.where(y_np == 1)[0]
    idx0 = np.where(y_np == 0)[0]
    ax1.scatter(idx1, [closes[i] for i in idx1], s=4, c='red', label='Label 1')
    ax1.scatter(idx0, [closes[i] for i in idx0], s=4, c='green', label='Label 0')
    ax1.set_ylabel('Close')
    ax1.set_title(f'{stock_code} Close/Label (top) vs GCN Avg Pseudo-attention (bottom) - Training Data')
    im2 = ax2.imshow(mean_mat, aspect='auto', origin='lower', cmap=cmap, interpolation='nearest')
    ax2.set_ylabel('day')
    ax2.set_yticks(lag_ticks)
    ax2.set_yticklabels([str(l + 1) for l in lag_ticks])
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([dates[p] for p in tick_pos], rotation=30)
    fig2.colorbar(im2, ax=[ax1, ax2], pad=0.01)
    
    # 保存和显示
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    f1_name = f'{ts}_{mode}_GCN_pseudo_attention_heatmap_layers_{stock_code}.png'
    f2_name = f'{ts}_{mode}_GCN_pseudo_attention_heatmap_price_aligned_{stock_code}.png'
    fig1.savefig(f1_name, dpi=150)
    fig2.savefig(f2_name, dpi=150)
    print(f'GCN pseudo-attention heatmaps saved: {f1_name} / {f2_name}')
    plt.show()

# 记录和打印训练/验证进度
def log_training_progress(epoch, loss, model, data, train_mask, val_mask, trainingTimes, printInterval=50, best_f1=0, quiet=False):
    """
    计算训练/验证指标并格式化输出（quiet模式下跳过训练集指标计算，加速搜索）
    :return: precision_val, recall_val, f1_val, best_f1
    """
    model.eval()
    with torch.no_grad():
        out_val = model(data.x, data.edge_index)
        # 验证集指标（始终需要，早停依赖验证F1）
        predicted_val = torch.argmax(out_val[val_mask], dim=1)
        p_val, r_val, f1_val, _ = precision_recall_fscore_support(data.y[val_mask].cpu(), predicted_val.cpu(), average='macro')
        acc_val = accuracy_score(data.y[val_mask].cpu(), predicted_val.cpu())
        # 训练中间过程（仅训练单个股票时打印中间过程，批量训练阶段不打印中间过程，省略不算）
        if not quiet and printInterval > 0 and ((epoch + 1) % printInterval == 0 or epoch == 0):
            predicted_tr = torch.argmax(out_val[train_mask], dim=1)
            p_tr, r_tr, f1_tr, _ = precision_recall_fscore_support(data.y[train_mask].cpu(), predicted_tr.cpu(), average='macro')
            acc_tr = accuracy_score(data.y[train_mask].cpu(), predicted_tr.cpu())
        else:
            p_tr = r_tr = f1_tr = acc_tr = 0.0

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
def create_scheduler(optimizer, ifOpen, patience=100, factor=0.5, min_lr=1e-5):
    """
    创建学习率调度器，返回None表示不启用
    """
    if not ifOpen:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=factor, patience=patience, min_lr=min_lr)

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

# 定义网络架构
class Net(torch.nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: 超参数字典（dropoutRate/ifOpenBatchNorm/residualHistoryN/ifOpenEdgeDropout/edgeDropoutRate等）
        """
        super(Net, self).__init__()
        # add_self_loops=False：确保预测第N天时只使用前N-1天数据，节点i只聚合邻居i-1的特征
        # 10层网络：5个Block，每2层一个Block，维度平滑过渡 7→32→32→64→64→128→128→128→128→64→2
        # 网络结构模式：mixed(GCN-GAT交替)/onlyGCN(全GCN)/onlyGAT(全GAT)，消融实验用
        self.netMode = cfg.get('netMode', 'mixed')
        # 10层维度配置：(in_dim, out_dim)，三种模式维度完全一致，仅层类型不同
        dims = [(7, 32), (32, 32), (32, 64), (64, 64), (64, 128),
                (128, 128), (128, 128), (128, 128), (128, 64), (64, 2)]
        # 记录每层是否为GAT（用于注意力收集：仅GAT层可返回attention权重）
        self.is_gat = []
        for i, (in_d, out_d) in enumerate(dims):
            if self.netMode == 'onlyGCN':
                layer = GCNConv(in_d, out_d, add_self_loops=False)
                self.is_gat.append(False)
            elif self.netMode == 'onlyGAT':
                layer = GATConv(in_d, out_d, add_self_loops=False)
                self.is_gat.append(True)
            else:  # mixed: 奇数层(1,3,5,7,9)=GCN, 偶数层(2,4,6,8,10)=GAT
                if i % 2 == 0:
                    layer = GCNConv(in_d, out_d, add_self_loops=False)
                    self.is_gat.append(False)
                else:
                    layer = GATConv(in_d, out_d, add_self_loops=False)
                    self.is_gat.append(True)
            setattr(self, f'conv{i+1}', layer)
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

    # 卷积层统一调用封装：仅当该层为GAT且需要收集注意力时，额外返回attention权重(edge_index, alpha)
    # is_gat标识当前层类型（onlyGCN全False/onlyGAT全True/mixed奇False偶True），保证三种模式forward路径一致
    def _call_conv(self, conv, x, edge_index, att_list, is_gat):
        if is_gat and att_list is not None:
            out, att = conv(x, edge_index, return_attention_weights=True)
            att_list.append(att)
            return out
        return conv(x, edge_index)

    def forward(self, x, edge_index, return_attention=False):
        # return_attention=True时额外返回5个GAT层的注意力权重列表（仅可视化时用，训练路径不受影响）
        att_list = [] if return_attention else None
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
        x = self._call_conv(self.conv1, x, edge_index, att_list, self.is_gat[0])
        if self.ifOpenBatchNorm: x = self.bn1(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip1 = x  # conv1输出(32维)，供Block1跨层残差使用
        # conv2: 短残差(32→32直接相加) + 跨层残差(conv1输出→conv2输出, 32→32直接相加)
        res = x
        x = self._call_conv(self.conv2, x, edge_index, att_list, self.is_gat[1])
        if self.ifOpenBatchNorm: x = self.bn2(x)
        x = F.relu(x + res + skip1)
        x = self.dropout(x)
        # === Block 2: conv3 + conv4（64维平台，含跨层残差） ===
        # conv3: 短残差
        res = self.proj3(x)
        x = self._call_conv(self.conv3, x, edge_index, att_list, self.is_gat[2])
        if self.ifOpenBatchNorm: x = self.bn3(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip3 = x  # conv3输出(64维)，供Block2跨层残差使用
        # conv4: 短残差(64→64直接相加) + 跨层残差(conv3输出→conv4输出, 64→64直接相加)
        res = x
        x = self._call_conv(self.conv4, x, edge_index, att_list, self.is_gat[3])
        if self.ifOpenBatchNorm: x = self.bn4(x)
        x = F.relu(x + res + skip3)
        x = self.dropout(x)
        # === Block 3: conv5 + conv6（128维平台，含跨层残差） ===
        # conv5: 短残差
        res = self.proj5(x)
        x = self._call_conv(self.conv5, x, edge_index, att_list, self.is_gat[4])
        if self.ifOpenBatchNorm: x = self.bn5(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip5 = x  # conv5输出(128维)，供Block3跨层残差使用
        # conv6: 短残差(128→128直接相加) + 跨层残差(conv5输出→conv6输出, 128→128直接相加)
        res = x
        x = self._call_conv(self.conv6, x, edge_index, att_list, self.is_gat[5])
        if self.ifOpenBatchNorm: x = self.bn6(x)
        x = F.relu(x + res + skip5)
        x = self.dropout(x)
        # === Block 4: conv7 + conv8（128维平台，含跨层残差） ===
        # conv7: 短残差(128→128直接相加)
        res = x
        x = self._call_conv(self.conv7, x, edge_index, att_list, self.is_gat[6])
        if self.ifOpenBatchNorm: x = self.bn7(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        skip7 = x  # conv7输出(128维)，供Block4跨层残差使用
        # conv8: 短残差 + 跨层残差(conv7输出→conv8输出, 128→128直接相加)
        res = x
        x = self._call_conv(self.conv8, x, edge_index, att_list, self.is_gat[7])
        if self.ifOpenBatchNorm: x = self.bn8(x)
        x = F.relu(x + res + skip7)
        x = self.dropout(x)
        # === Block 5: conv9 + conv10（降维+输出） ===
        # conv9: 短残差
        res = self.proj9(x)
        x = self._call_conv(self.conv9, x, edge_index, att_list, self.is_gat[8])
        if self.ifOpenBatchNorm: x = self.bn9(x)
        x = F.relu(x + res)
        x = self.dropout(x)
        # conv10: 输出层，不加残差
        x = self._call_conv(self.conv10, x, edge_index, att_list, self.is_gat[9])
        out = F.log_softmax(x, dim=1)
        if return_attention:
            return out, att_list
        return out

# 单/多股票建图：每只股票按cfg的K/stride独立建图，再拼成一张大图（跨股票无边相连，信息不跨股票流动）
def build_graph(stock_data_list, cfg):
    """
    单/多股票建图：每只股票独立建图后拼成大图，归一化并预转换类型，list中若只有一个股票即为单股票训练模式
    :return: (data, train_mask, val_mask, test_mask)
    """
    data_list = []
    train_mask, val_mask, test_mask = [], [], []
    for priceDic, tr_mask, va_mask, te_mask, code in stock_data_list:
        d = TrainData.TrainDataMACDWindowK(priceDic, cfg['edgeWindowK'], cfg['edgeStride'])[0]
        data_list.append(d)
        train_mask.extend(tr_mask)
        val_mask.extend(va_mask)
        test_mask.extend(te_mask)
    data = Batch.from_data_list(data_list)
    if cfg['ifOpenNormalize']:
        data = normalize_features(data, train_mask)
    data = data.to(device)
    data.x = data.x.to(torch.float32)
    data.y = data.y.to(torch.long)
    return data, train_mask, val_mask, test_mask

# 测试集评估：在测试集上计算多项分类指标
def evaluate_test(model, data, test_mask):
    """
    测试集评估，返回指标字典
    :return: dict(accuracy/precision/recall/f1/cm)
    """
    model.eval()
    with torch.no_grad():
        test_predict = model(data.x, data.edge_index)[test_mask]
        max_index = torch.argmax(test_predict, dim=1)
        test_true = data.y[test_mask]
    test_pred = max_index.cpu().numpy()
    test_true_np = test_true.cpu().numpy()
    accuracy = accuracy_score(test_true_np, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true_np, test_pred, average='macro')
    cm = confusion_matrix(test_true_np, test_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

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
    data, train_mask, val_mask, test_mask = build_graph(stock_data_list, cfg)
    model = Net(cfg).to(device)
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
    scheduler = create_scheduler(optimizer, ifOpenLRScheduler, lrPatience, lrFactor, lrMinLr)
    focal_loss_fn = FocalLoss(alpha=class_weight_tensor, gamma=cfg['focalLossGamma']) if cfg['ifOpenFocalLoss'] else None
    if not quiet:
        print(f'本次训练配置: {cfg}')
        print(f'全局配置: 股票数={len(stock_data_list)}, 训练轮次={epochs}, 早停={ifOpenEarlyStop}(patience={cfg.get("earlyStopPatience", earlyStopPatience)}), 学习率调度={ifOpenLRScheduler}')
        if cfg['ifOpenFocalLoss']:
            print(f"Focal Loss已启用: gamma={cfg['focalLossGamma']}, alpha={class_weight_tensor}")
        if cfg['residualHistoryN'] > 1:
            print(f"短残差历史窗口: {cfg['residualHistoryN']}步拼接（维度 {7*cfg['residualHistoryN']}→32）")
        print(f"入边窗口: K={cfg['edgeWindowK']}, 稀疏间隔={cfg['edgeStride']}（每节点直接聚合前{cfg['edgeWindowK']}天内隔{cfg['edgeStride']}取一，边数={data.edge_index.shape[1]}）")

    precisions, recalls, f1s, losses = [], [], [], []
    # 初始化早停控制器
    early_stopper = EarlyStopper(cfg.get('earlyStopPatience', earlyStopPatience)) if ifOpenEarlyStop else None
    # 最佳F1初始化，用于记录训练过程中最佳验证F1及其出现轮次
    best_f1 = 0.0
    best_epoch = 0
    #模型训练/验证
    train_start = time.time()   #记录训练开始时间，用于统计耗时
    # 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)    #模型的输入有节点特征还有边特征,使用的是全部数据
        if focal_loss_fn is not None:
            loss = focal_loss_fn(out[train_mask], data.y[train_mask])
        else:
            loss = F.nll_loss(out[train_mask], data.y[train_mask], weight=class_weight_tensor)   #损失仅仅计算的是训练集的损失
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # 计算训练/验证指标并输出（quiet时printInterval=0不打印）
        prev_best_f1 = best_f1
        precision_val, recall_val, f1_val, best_f1 = log_training_progress(epoch, loss, model, data, train_mask, val_mask, epochs, 0 if quiet else printInterval, best_f1, quiet=quiet)
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
        #执行完model.eval()后重新开始train模式
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
    metrics = evaluate_test(model, data, test_mask)
    return {'best_val_f1': best_f1, 'accuracy': metrics['accuracy'], 'precision': metrics['precision'],
            'recall': metrics['recall'], 'f1': metrics['f1'], 'cm': metrics['cm'],
            'model': model, 'best_epoch': best_epoch, 'elapsed': train_elapsed,
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
def sample_configs(space, nTrials, ensure_baseline=True):
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
    # 强制包含单链基准组合（edgeWindowK=1且edgeStride=1，其余参数沿用第1组），作为窗口结构的对照基准
    if ensure_baseline and configs:
        configs[0] = {**configs[0], 'edgeWindowK': 1, 'edgeStride': 1}
    return configs

if __name__ == '__main__':
    lg = bs.login()
    # 遍历码表获取所有股票（每只股票内部按时序75/10/15划分train/val/test，后续拼成大图时各mask拼接）
    stock_data_list = []  # 每个元素: (priceDic, train_mask, val_mask, test_mask, code)

    if ifOpenMultiStock:
        # 多股票模式：遍历沪深300码表，每只股票独立处理后拼成大图
        stockPoolList = StockPool.GetHS300StockListBaostock()
        dataCount = 0
        for code in StockPool.GetALLStockListBaostock(getNewStockPoolByDate).keys():
            if len(stockPoolList) == 0 or code not in stockPoolList:
                continue
            if maxStockCount is not None and dataCount >= maxStockCount:
                break
            try:
                result = process_single_stock(code, dataDate, periodRange)
                if result is not None:
                    stock_data_list.append(result)
                    dataCount += 1
                    print(f'{code} 预处理完成,序号:NO.{dataCount},节点数:{len(result[0])}')
                else:
                    print(code + ' 数据不足或指标出错,跳过')
            except Exception as ex:
                print("失败代码："+code+"，异常信息："+str(ex))
        print(f'共预处理 {len(stock_data_list)} 只股票，总节点数 {sum(len(r[0]) for r in stock_data_list)}')
    else:
        # 单股票模式：仅用stockCode构建单只股票的图
        result = process_single_stock(stockCode, dataDate, periodRange)
        if result is not None:
            stock_data_list.append(result)
            print(f'{stockCode} 预处理完成,节点数:{len(result[0])}')
        else:
            print(f'{stockCode} 数据不足或指标出错')

    # 预处理结果检查：无可用数据时终止程序
    if len(stock_data_list) == 0:
        print('错误：没有成功预处理任何股票，程序终止')
        bs.logout()
        sys.exit(1)

    #主流程：消融实验模式 / 超参数搜索模式 / 单次训练模式
    # 基础配置（各模式共用，消融实验在此基础上覆盖netMode）
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
        'netMode': netMode,
    }
    if ifOpenAblation:
        # 消融实验模式：遍历ablationModes各组训练，输出对比表+热力图，量化GCN/GAT对训练的影响
        print(f'========== 消融实验：网络结构模式对比，共{len(ablationModes)}组 ==========')
        ablation_results = []
        for mode in ablationModes:
            print(f'\n----- 消融组 [{mode}] -----')
            cfg = {**base_cfg, 'netMode': mode}
            r = run_training(cfg, stock_data_list, quiet=False)
            ablation_results.append((mode, r, cfg))
        # 对比表
        print('\n========== 消融实验结果对比 ==========')
        print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
        print('-' * 75)
        for mode, r, _ in ablation_results:
            print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
        print('-' * 75)
        for mode, r, _ in ablation_results:
            print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
        # 热力图：统一调用plot_attention_heatmaps（GAT层用注意力权重，GCN层用伪注意力，全部10层显示）
        for mode, r, cfg in ablation_results:
            if cfg['edgeWindowK'] <= 1:
                print(f'[{mode}] edgeWindowK<=1，跳过热力图')
                continue
            vis_stock = stock_data_list[0]  # (priceDic, train_mask, val_mask, test_mask, code)
            data_vis, _, _, _ = build_graph([vis_stock], cfg)
            input(f'\n按回车键查看 [{mode}] 各层热力图...')
            plot_attention_heatmaps(r['model'], data_vis, vis_stock[0], cfg, vis_stock[1], vis_stock[2], vis_stock[4], mode=mode)
        bs.logout()
        sys.exit(0)
    elif ifOpenHyperSearch:
        # 搜索模式下屏蔽sklearn的UndefinedMetricWarning（某类无预测样本时的警告），避免刷屏干扰每组摘要行；单次训练模式不屏蔽
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        # 固定采样种子，保证搜索组合可复现（需在run_training重置种子前一次性采样完，采样与训练互不影响）
        set_seed(2)
        trial_configs = sample_configs(hyperSearchSpace, hyperSearchTrials)
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

    # GAT注意力热力图可视化（取第一只股票单独重建单股图，避免多股大图交界处滞后错乱；单股模式即stockCode本身）
    if ifOpenAttentionHeatmap:
        used_cfg = best_cfg if ifOpenHyperSearch else base_cfg
        if used_cfg['edgeWindowK'] <= 1:
            print('注意力热力图跳过：edgeWindowK<=1时每节点仅1条入边，softmax后注意力恒为1，无展示意义')
        else:
            vis_stock = stock_data_list[0]  # (priceDic, train_mask, val_mask, test_mask, code)
            data_vis, _, _, _ = build_graph([vis_stock], used_cfg)
            input('\n按回车键查看GAT注意力热力图...')
            plot_attention_heatmaps(result['model'], data_vis, vis_stock[0], used_cfg, vis_stock[1], vis_stock[2], vis_stock[4])

    # 训练过程参数变化可视化（按回车后显示图表）
    #input('\n按回车键查看训练指标曲线图...')
    #plot_metrics(result['precisions'], result['recalls'], result['f1s'], result['losses'])
    bs.logout()

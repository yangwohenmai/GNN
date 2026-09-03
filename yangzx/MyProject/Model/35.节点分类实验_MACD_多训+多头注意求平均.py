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
import signal
import random
import time
import traceback
import warnings
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, precision_score, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
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
stockCode = '000001.SZ'
dataDate = "20260101"       # 训练数据取值范围的截止日期
periodRange = 1400          # 根据dataDate，向前取多少个自然日
# 获取最新日期，取出当天所有股票作为股票池（默认取周一的股票池）
getNewStockPoolByDate = datetime.fromordinal(datetime.today().toordinal() - (datetime.today().weekday() or 7)).strftime('%Y-%m-%d')
useLocalData = False         # 是否使用本地缓存数据（True=从本地txt文件读取行情和股票池，False=联网从baostock获取）
ifOpenMultiStock = True     # 是否启用多股票训练（True=遍历沪深300码表拼大图，False=仅用stockCode单股票训练）
maxStockCount = 30          # 用多少只股票同时训练（仅多股票模式生效，None=不限制）
dropoutRate = 0.1           # Dropout率
trainingTimes = 20000        # 训练轮次
printInterval = 50          # 训练参数打印间隔
ifOpenNormalize = True      # 是否启用归一化（不开）
ifOpenEarlyStop = True      # 是否启用早停（不开）
earlyStopPatience = 2000     # 连续多少轮验证F1未提升则停止
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
residualHistoryN = 5        # conv1历史注入窗口大小（1=仅x[i-1]，n=前n个历史节点x[i-n]~x[i-1]拼接后投影注入；与抗梯度消失的残差无关）
edgeWindowK =21             # 入边窗口大小（每个节点i接收前K个相邻节点的边X[i-K]~X[i-1]→X[i]，1=单链结构）
edgeStride = 3              # 入边稀疏间隔（从X[i-1]开始每隔stride取一个，如K=3、stride=2时仅X[i-3]、X[i-1]→X[i]，1=稠密窗口）
numAttentionHeads = 1       # GAT注意力头数（1=单头；超参数搜索时此值被搜索空间覆盖，非搜索路径用此值）
ifOpenGATConcat = False     # GAT多头注意力输出方式（False=多头取平均，维度不变；True=多头拼接，维度=heads*out_d）
ifOpenAttentionHeatmap = True  # 是否在训练结束后绘制GAT层热力图（需edgeWindowK>1才有意义，K=1时每节点仅1条入边注意力恒为1）
ifOpenAblation = False       # 是否启用消融实验模式（开启后遍历ablationModes各组训练并输出对比表，量化GCN/GAT对训练的影响）
ablationModes = ['mixed', 'onlyGCN', 'onlyGAT']  # 消融实验对比的网络模式列表（mixed=当前GCN-GAT交替基准）
ifOpenHyperSearch = True    # 是否启用超参数随机搜索（开启后搜索空间内参数的全局值失效，自动寻找最佳组合）
hyperSearchTrials = 30      # 随机搜索采样组数
hyperSearchTrainingTimes = 3000  #搜索阶段每组训练轮次（短轮次快速筛选，选出最佳组合后再用trainingTimes完整训练）
hyperSearchSpace = {        # 搜索空间：参数名→候选值列表（可自行增删候选值）
    'ifOpenNormalize':   [True],            #[True, False],
    'ifOpenClassWeight': [False],           #[True, False],
    'ifOpenBatchNorm':   [False],           #[True, False],
    'residualHistoryN':  [1, 3, 5],
    'edgeWindowK':       [1, 5, 9, 15],
    'edgeStride':        [1, 2, 3, 4],
    'dropoutRate':       [0.1, 0.2, 0.3],
    'ifOpenEdgeDropout': [False],           #[True, False],
    'edgeDropoutRate':   [0.2],             #[0.1, 0.2, 0.3],
    'ifOpenFocalLoss':   [False],           #[True, False],
    'focalLossGamma':    [1.0],             #[1.0, 2.0],
    'earlyStopPatience': [200],             #[50, 100, 200]搜索阶段用小patience加速（单次训练模式用全局earlyStopPatience=800）
    'numAttentionHeads': [2, 3, 4],         # GAT注意力头数（1=单头，2/4=多头取平均，维度不变）
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #运行设备：有GPU用cuda，否则用cpu
modelSaveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')  # 模型保存目录（方式三保存/方式四加载共用）

# 中断恢复：记录当前处理的股票代码（信号处理器用）
current_code = None
allStockSorted = []  # 当前遍历的股票列表（信号处理器用）

# 训练中断：记录当前训练上下文（信号处理器用，用于Ctrl+C时保存最佳模型）
current_training_context = None  # {'model': model, 'early_stopper': early_stopper, 'cfg': cfg, 'dataDate': dataDate, 'stock_count': stock_count}


# 获取股票池列表（根据useLocalData开关自动选择本地缓存或联网获取）
# indexCode: 'hs300'=沪深300，'all'=全市场
def get_stock_pool_list(indexCode='hs300'):
    if useLocalData:
        result = StockData.GetStockPoolListFromTxt(indexCode)
        if result == False:
            print(f'错误：本地缓存中无{indexCode}股票池，请先运行 StockData.DownloadStockPoolList() 下载')
            sys.exit(1)
        return result
    else:
        if indexCode == 'hs300':
            return StockPool.GetHS300StockListBaostock()
        elif indexCode == 'all':
            return StockPool.GetALLStockListBaostock(getNewStockPoolByDate)
        else:
            return {}

# 获取股票行情数据（根据useLocalData开关自动选择本地缓存或联网获取）
def get_stock_price(code, endDate, period):
    if useLocalData:
        return StockData.GetStockPriceFromTxt(code, endDate, period)
    else:
        return StockData.GetStockPriceDWMBaostock(code, endDate, period)

# 校验本地缓存数据是否覆盖endDate（仅useLocalData=True时调用，实盘/回测通用）
# 规则：
#   1. endDate为周末时基准回退到上周五并提示（非交易日无数据属正常）；
#      endDate落在节假日（工作日休市）时校验自然不通过，提示用户改日期
#   2. 逐只轻量读取缓存文件末尾的最后日期，文件不存在或最后日期早于基准 → 计为缺失（少量缺失不影响整体）
#   3. 缺失占比严格小于 maxMissingRatio（20%）则放行，否则返回False阻止继续
def check_local_data_coverage(stockCodes, endDateStr, maxMissingRatio=0.2):
    end_dt = datetime.strptime(endDateStr, '%Y%m%d')
    check_dt = end_dt
    if end_dt.weekday() == 5:      # 周六 → 上周五
        check_dt = end_dt - timedelta(days=1)
        print(f'提示：{endDateStr} 是周六，按最近交易日 {check_dt.strftime("%Y-%m-%d")} 校验')
    elif end_dt.weekday() == 6:    # 周日 → 上周五
        check_dt = end_dt - timedelta(days=2)
        print(f'提示：{endDateStr} 是周日，按最近交易日 {check_dt.strftime("%Y-%m-%d")} 校验')
    check_date_str = check_dt.strftime('%Y-%m-%d')

    cacheDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    missing_list = []       # (代码, 缓存最后日期)
    overall_latest = None   # 缓存中见到的最新日期（用于报警信息）
    for code in stockCodes:
        # 轻量读取缓存文件末尾1KB取最后交易日期，不加载全部数据；文件不存在计为缺失
        last_date = None
        filepath = os.path.join(cacheDir, f'{code}.txt')
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    f.seek(max(0, f.tell() - 1024))
                    chunk = f.read().decode('utf-8', errors='ignore')
                lines = [l for l in chunk.strip().splitlines() if l.strip()]
                if lines:
                    parts = lines[-1].strip().split(',')
                    if len(parts) > 1:
                        last_date = parts[1]
            except Exception:
                last_date = None
        if last_date is not None and (overall_latest is None or last_date > overall_latest):
            overall_latest = last_date
        if last_date is None or last_date < check_date_str:  # 'YYYY-MM-DD' 格式可按字典序比较
            missing_list.append((code, last_date))

    if len(missing_list) == 0:
        return True, ''
    # 容忍条件：缺失占比严格小于20%（用整数运算 缺失数*5 < 总数 判断，避免浮点误差）
    if len(missing_list) * 5 < len(stockCodes):
        sample = ', '.join([f'{c}({d if d else "无文件"})' for c, d in missing_list[:10]])
        print(f'提示：{len(missing_list)} 只股票本地数据未到 {check_date_str}（疑似停牌/新股，占比低于20%，不影响整体）：{sample}')
        return True, ''

    examples = ', '.join([f'{c}({d if d else "无文件"})' for c, d in missing_list[:5]])
    missing_ratio_str = f'{len(missing_list) / len(stockCodes) * 100:.1f}%'
    msg = ('\n========== 本地数据覆盖校验失败，已停止 =========='
           f'\n请求数据截止：{endDateStr}（校验基准：{check_date_str}）'
           f'\n共检查 {len(stockCodes)} 只，缺失 {len(missing_list)} 只，占比 {missing_ratio_str}（容忍上限：严格小于20%）'
           f'\n缓存中最新数据日期：{overall_latest if overall_latest else "无可用数据"}'
           f'\n缺失示例：{examples} ...'
           '\n可能原因：本地缓存过期，或该日为节假日无交易'
           '\n处理建议：'
           f'\n  1) 运行 StockData.UpdateStockDataByDate(...) 把数据补到 {check_date_str}'
           '\n  2) 或改为 useLocalData=False 联网获取'
           '\n  3) 若该日确为节假日，请把 dataDate 改为最近交易日')
    return False, msg

# 股票预处理：每只股票独立处理（行情→BLJJ→flag→过滤→mask），收集后供run_training拼接成大图
def process_single_stock(code, endDate, period=1400):
    """
    处理单只股票：拉行情 → BLJJ → flag标注 → 过滤空窗 → mask构建
    :return: (priceDic, train_mask, val_mask, test_mask, code) 或 None（失败时）
    """
    stockPriceDic = get_stock_price(code, endDate, period)
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
    # 标签前移一天后，末节点的标签是"次日flag"（未来未知，在TrainData中填-1），
    # 必须同时从训练/验证/测试三个集合里排除，否则-1会进入损失和评估
    if len(train_mask) > 0:
        train_mask[-1] = False
        val_mask[-1] = False
        test_mask[-1] = False
    return newStockPriceDic, train_mask, val_mask, test_mask, code

# 特征标准化（仅用训练集统计量，防止测试集信息泄露）
def normalize_features(data, train_mask, scaler=None):
    """
    对节点特征做标准化，消除量纲差异（仅 fit 训练集，防止数据泄露）
    :param data: PyG Data 对象
    :param train_mask: 训练集 mask（list[bool]）
    :param scaler: 如果提供，则使用该scaler进行transform；否则fit一个新的scaler
    :return: (data, scaler)（原地修改后返回data和scaler）
    """
    x_np = data.x.numpy().astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x_np[train_mask, :6])            # 只用训练集 fit 前6列（open/close/low/high/pctChg/X轴位置；第7列flag是标签不标准化）
    x_np[:, :6] = scaler.transform(x_np[:, :6]) # transform 全部数据
    data.x = torch.tensor(x_np, dtype=torch.float32)
    return data, scaler

#region 固定所有随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#endregion

#region ========== 日志写入函数（统一格式，追加写入当前脚本同目录） ==========
def log_comparison_result(stockCode, compare_results, filename='对比结果.txt'):
    """
    将多模式对比结果格式化追加写入日志文件
    :param stockCode: 股票代码
    :param compare_results: list of (mode, result_dict, cfg)
    :param filename: 日志文件名，默认'对比结果.txt'
    """
    lines = []
    lines.append(f'\n{"=" * 60}\n')
    lines.append(f'对比结果 ({stockCode})  运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    lines.append(f'{"=" * 60}\n')
    lines.append(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}\n')
    lines.append('-' * 75 + '\n')
    for mode, r, _ in compare_results:
        lines.append(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}\n')
    lines.append('-' * 75 + '\n')
    for mode, r, _ in compare_results:
        lines.append(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n\n')
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(''.join(lines))

#endregion

#region 错误日志
def log_error(stockCode, error_msg, filename='error.txt'):
    """
    将错误信息格式化追加写入错误日志文件
    :param stockCode: 股票代码
    :param error_msg: 错误信息（字符串，可以是traceback或简短描述）
    :param filename: 错误日志文件名，默认'error.txt'
    """
    lines = []
    lines.append(f'\n{"=" * 60}\n')
    lines.append(f'股票: {stockCode}  时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    lines.append(f'{"=" * 60}\n')
    lines.append(f'{error_msg}\n')
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(''.join(lines))
#endregion

#region ========== 控制台↔txt同步日志 ==========
_active_log_file = None  # 当前打开的日志文件句柄（run_all_func_lite入口打开，未开启日志时为None）

def log_print(msg=''):
    """
    控制台与txt日志双输出，内容格式完全一致（原样写入）
    仅当_active_log_file已打开时写文件（由run_all_func_lite开启），其余场景行为等同print
    """
    print(msg)
    if _active_log_file is not None:
        _active_log_file.write(str(msg) + '\n')
        _active_log_file.flush()
#endregion

#region ========== 信号处理器：捕获 Ctrl+C 时记录中断点 ==========
def signal_handler(signum, frame):
    """
    程序被 Ctrl+C 或 kill 命令终止时的清理函数
    自动记录当前处理的股票代码，并建议下次 resume_from 的值
    如果正在训练，保存最佳模型
    """
    sig_name = signal.Signals(signum).name
    msg = f'程序被手动中断\n信号: {sig_name}\n当前处理到: {current_code}'
    if current_code and allStockSorted:
        # 计算下一个代码
        try:
            idx = allStockSorted.index(current_code)
            if idx + 1 < len(allStockSorted):
                next_code = allStockSorted[idx + 1]
                msg += f'\n下次 resume_from 设为: {next_code}'
        except:
            pass
    log_error('中断日志', msg)
    print(f'\n程序已中断，中断信息已写入 error.txt')
    
    # Ctrl+C 中断时保存模型
    if current_training_context is not None:
        ctx = current_training_context
        early_stopper = ctx.get('early_stopper')
        model = ctx.get('model')
        cfg = ctx.get('cfg')
        dataDate = ctx.get('dataDate')
        stock_count = ctx.get('stock_count')
        periodRange = ctx.get('periodRange')
        edgeWindowK = ctx.get('edgeWindowK')
        edgeStride = ctx.get('edgeStride')
        residualHistoryN = ctx.get('residualHistoryN')
        
        if model is not None:
            print(f'\n检测到正在训练，保存最佳模型...')
            # 如果早停器记录了最佳状态（验证集F1最高），加载该状态
            if early_stopper is not None and early_stopper.best_state is not None:
                model.load_state_dict(early_stopper.best_state)
            # 计算当前模型在测试集上的准确率，用于文件名标注
            data = ctx.get('data')
            test_mask = ctx.get('test_mask')
            current_acc = None
            if data is not None and test_mask is not None:
                try:
                    metrics = evaluate_test(model, data, test_mask)
                    current_acc = metrics['accuracy']
                except Exception:
                    pass
            mode = cfg.get('netMode', 'unknown')
            save_trained_model(model, dataDate, mode, modelSaveDir, ctx.get('scaler'), stock_count, current_acc, 'stop', periodRange, edgeWindowK, edgeStride, residualHistoryN, cfg)
            if current_acc is not None:
                print(f'模型已保存（测试Acc={current_acc*100:.2f}%）')
            else:
                print(f'模型已保存')
    
    # 登出 baostock（防止中断时 baostock 会话泄漏）
    try:
        bs.logout()
    except:
        pass
    sys.exit(0)
#endregion

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

# GAT层热力图：滞后lag×时间，展示“预测第i天时对前K天历史的注意力分配”
# 注：自环也占一份注意力，但其lag=0不在本图展示范围内，
#     因此各列之和小于1（差额即模型分给当天的注意力）
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
        _, att_list = model(data.x, data.edge_index, batch=data.batch, return_attention=True)
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
        out_val = model(data.x, data.edge_index, batch=data.batch)
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

def convert_gat_state_dict(state_dict, model_state):
    """
    兼容不同PyG版本的GATConv参数命名（lin vs lin_src/lin_dst）
    :param state_dict: 从checkpoint加载的state_dict
    :param model_state: 当前模型的state_dict
    :return: 转换后的state_dict
    """
    saved_has_lin = any('lin.weight' in k for k in state_dict.keys())
    saved_has_lin_src = any('lin_src.weight' in k for k in state_dict.keys())
    model_has_lin = any('lin.weight' in k for k in model_state.keys())
    model_has_lin_src = any('lin_src.weight' in k for k in model_state.keys())
    
    if saved_has_lin and model_has_lin_src:
        # 保存的模型用lin，当前模型用lin_src/lin_dst：复制lin到两者
        print('检测到PyG版本差异，正在转换模型参数 (lin → lin_src/lin_dst)...')
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'lin.weight' in k:
                new_state_dict[k.replace('lin.weight', 'lin_src.weight')] = v
                new_state_dict[k.replace('lin.weight', 'lin_dst.weight')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    elif saved_has_lin_src and model_has_lin:
        # 保存的模型用lin_src/lin_dst，当前模型用lin：取平均
        print('检测到PyG版本差异，正在转换模型参数 (lin_src/lin_dst → lin)...')
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'lin_src.weight' in k:
                lin_dst_key = k.replace('lin_src', 'lin_dst')
                if lin_dst_key in state_dict:
                    new_key = k.replace('lin_src.weight', 'lin.weight')
                    new_state_dict[new_key] = (v + state_dict[lin_dst_key]) / 2
            elif 'lin_dst.weight' in k:
                continue  # 已处理
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    return state_dict

# 定义网络架构
class Net(torch.nn.Module):
    def __init__(self, cfg):
        """
        :param cfg: 超参数字典（dropoutRate/ifOpenBatchNorm/residualHistoryN/ifOpenEdgeDropout/edgeDropoutRate等）
        """
        super(Net, self).__init__()
        # 输入特征维度：优先从 cfg['featDim'] 动态获取（由建图后的数据决定），
        # 缺省回退7（兼容旧checkpoint/未注入featDim的cfg）
        feat_dim = cfg.get('featDim', 7)
        self.feat_dim = feat_dim
        # 10层网络：5个Block，每2层一个Block，维度平滑过渡 feat→32→32→64→64→128→128→128→128→64→2
        # 网络结构模式：mixed(GCN-GAT交替)/onlyGCN(全GCN)/onlyGAT(全GAT)，消融实验用
        self.netMode = cfg.get('netMode', 'mixed')
        # 10层维度配置：(in_dim, out_dim)，三种模式维度完全一致，仅层类型不同（首层输入维度随特征列数动态变化）
        dims = [(feat_dim, 32), (32, 32), (32, 64), (64, 64), (64, 128),
                (128, 128), (128, 128), (128, 128), (128, 64), (64, 2)]
        # 记录每层是否为GAT（用于注意力收集：仅GAT层可返回attention权重）
        self.is_gat = []
        for i, (in_d, out_d) in enumerate(dims):
            if self.netMode == 'onlyGCN':
                layer = GCNConv(in_d, out_d)
                self.is_gat.append(False)
            elif self.netMode == 'onlyGAT':
                layer = GATConv(in_d, out_d, heads=cfg.get('numAttentionHeads', 1), concat=ifOpenGATConcat)
                self.is_gat.append(True)
            else:  # mixed: 奇数层(1,3,5,7,9)=GCN, 偶数层(2,4,6,8,10)=GAT
                if i % 2 == 0:
                    layer = GCNConv(in_d, out_d)
                    self.is_gat.append(False)
                else:
                    layer = GATConv(in_d, out_d, heads=cfg.get('numAttentionHeads', 1), concat=ifOpenGATConcat)
                    self.is_gat.append(True)
            setattr(self, f'conv{i+1}', layer)
        self.dropout = torch.nn.Dropout(cfg['dropoutRate'])
        self.edge_dropout_rate = cfg['edgeDropoutRate'] if cfg['ifOpenEdgeDropout'] else 0.0
        self.residualHistoryN = cfg['residualHistoryN']
        # proj1为conv1的历史信息注入通道（非残差：输入源是历史节点特征而非本层输入，不抗梯度消失）
        # proj3/5/9为真正的残差投影层（维度不匹配时做线性投影对齐，近似恒等捷径）
        # residualHistoryN=1时输入feat_dim维；n>1时拼接n个历史节点特征，输入feat_dim*n维
        self.proj1 = torch.nn.Linear(feat_dim * cfg['residualHistoryN'], 32)    # conv1历史注入（前n个历史节点特征拼接后投影）
        self.proj3 = torch.nn.Linear(32, 64)   # conv3残差投影（32→64）
        self.proj5 = torch.nn.Linear(64, 128)  # conv5残差投影（64→128）
        self.proj9 = torch.nn.Linear(128, 64)  # conv9残差投影（128→64）
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

    def forward(self, x, edge_index, batch=None, return_attention=False):
        # return_attention=True时额外返回5个GAT层的注意力权重列表（仅可视化时用，训练路径不受影响）
        att_list = [] if return_attention else None
        #训练时随机丢弃边，防止过度依赖特定邻居。
        #安全性：自环不会误删——建图产生的edge_index本身不含自环，
        #自环由GATConv/GCNConv在forward内部添加（晚于dropout_edge，已用p=1.0极端用例验证）
        if self.training and self.edge_dropout_rate > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_rate)
        # === Block 1: conv1 + conv2（32维平台，含跨层残差） ===
        # conv1: 历史信息注入（非残差）——前residualHistoryN个历史节点的特征拼接后经proj1投影，与conv1输出相加。
        # 作用：给节点一条"连续N天历史特征"的直连数据通道，与边窗口的稀疏注意力聚合互补；
        # 输入源是历史特征而非本层输入，不提供恒等梯度捷径，不承担抗梯度消失职责。
        # 历史范围用shift排除当日x[i]，是标签平移前的防泄露遗留设计；
        # 注：标签已前移一天，当日x[i]不再是答案，卷积层经自环可看x[i]，注入通道仍保守地只用历史
        # n=1时: shifted_x[i] = x[i-1]（当前行为）
        # n>1时: 拼接 x[i-n], x[i-n+1], ..., x[i-1]（缺失位置补零向量）
        # 多股票模式：按段内shift，避免跨股票泄漏（batch=None时退化为全局shift）
        # 注意：不能用 shifted_k[mask][k:]=... 链式索引赋值（布尔索引返回副本，赋值不生效）
        shifted_list = []
        for k in range(self.residualHistoryN, 0, -1):
            if batch is not None:
                # 按股票分段shift：段内节点取本段前k个位置的特征，段首k个节点补零
                segs = []
                for seg_id in range(int(batch.max().item()) + 1):
                    seg_x = x[batch == seg_id]
                    if seg_x.size(0) > k:
                        pad = torch.zeros(k, x.size(1), device=x.device, dtype=x.dtype)
                        segs.append(torch.cat([pad, seg_x[:-k]], dim=0))
                    else:
                        segs.append(torch.zeros_like(seg_x))
                shifted_k = torch.cat(segs, dim=0)
            else:
                # 单股票：全局shift
                shifted_k = torch.zeros_like(x)
                shifted_k[k:] = x[:-k]
            shifted_list.append(shifted_k)
        shifted_x = torch.cat(shifted_list, dim=1)
        res = self.proj1(shifted_x)  # 历史注入项（非恒等残差）
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
def build_graph(stock_data_list, cfg, scaler=None):
    """
    单/多股票建图：每只股票独立建图后拼成大图，归一化并预转换类型，list中若只有一个股票即为单股票训练模式
    :param scaler: 如果提供，则使用该scaler进行归一化；否则fit一个新的scaler
    :return: (data, train_mask, val_mask, test_mask, scaler)
    """
    data_list = []
    train_mask, val_mask, test_mask = [], [], []
    for priceDic, tr_mask, va_mask, te_mask, code in stock_data_list:
        d = TrainData.TrainDataMACDWindowK_NextDay(priceDic, cfg['edgeWindowK'], cfg['edgeStride'])[0]
        data_list.append(d)
        train_mask.extend(tr_mask)
        val_mask.extend(va_mask)
        test_mask.extend(te_mask)
    data = Batch.from_data_list(data_list)
    if cfg['ifOpenNormalize']:
        data, scaler = normalize_features(data, train_mask, scaler)
    data = data.to(device)
    data.x = data.x.to(torch.float32)
    data.y = data.y.to(torch.long)
    return data, train_mask, val_mask, test_mask, scaler

# 测试集评估：在测试集上计算多项分类指标
def evaluate_test(model, data, test_mask):
    """
    测试集评估，返回指标字典
    :return: dict(accuracy/precision/recall/f1/cm)
    """
    model.eval()
    with torch.no_grad():
        test_predict = model(data.x, data.edge_index, batch=data.batch)[test_mask]
        max_index = torch.argmax(test_predict, dim=1)
        test_true = data.y[test_mask]
    test_pred = max_index.cpu().numpy()
    test_true_np = test_true.cpu().numpy()
    accuracy = accuracy_score(test_true_np, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true_np, test_pred, average='macro')
    cm = confusion_matrix(test_true_np, test_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'cm': cm}

#region ========== 模型保存/加载 ==========
def save_trained_model(model, dataDate, mode, save_dir, scaler=None, stock_count=None, accuracy=None, stop_prefix='', periodRange=None, edgeWindowK=None, edgeStride=None, residualHistoryN=None, cfg=None):
    """
    保存模型权重和归一化参数
    文件名格式：{dataDate}_{mode}[_{stock_count}s]_{config}_{accuracy}.pth
    示例：
      - 逐股训练：20260821_000001.SZ_onlyGAT_(1400d,21,3,5r)_72.34.pth
      - 拼大图训练：20260821_onlyGAT_300s_(1400d,21,3,5r)_75.61.pth
      - 中断保存：20260821_onlyGAT_300s_(1400d,21,3,5r)_stop68.52.pth
    
    :param model: 训练好的模型
    :param dataDate: 数据日期（如 '20250101'）
    :param mode: 网络模式名（如 'mixed'、'onlyGAT'、'000001.SZ_onlyGAT'）
    :param save_dir: 保存目录
    :param scaler: 归一化参数（StandardScaler对象）
    :param stock_count: 训练股票数量（如 300），文件名显示为 300s
    :param accuracy: 测试集准确率（如 0.7561），文件名显示为 75.61
    :param stop_prefix: 中断保存时的前缀（如 'stop'），文件名显示为 stop72.34
    :param periodRange: 数据周期范围（如 1400），文件名显示为 1400d
    :param edgeWindowK: 入边窗口大小（如 21）
    :param edgeStride: 入边稀疏间隔（如 3）
    :param residualHistoryN: 残差历史步数（如 5），文件名显示为 5r
    """
    os.makedirs(save_dir, exist_ok=True)
    # 拼接文件名各部分
    parts = [dataDate, mode]
    if stock_count is not None:
        parts.append(f'{stock_count}s')
    # 拼接数据配置参数：(1400d,21,3,5r)
    config_parts = []
    if periodRange is not None:
        config_parts.append(f'{periodRange}d')
    if edgeWindowK is not None:
        config_parts.append(str(edgeWindowK))
    if edgeStride is not None:
        config_parts.append(str(edgeStride))
    if residualHistoryN is not None:
        config_parts.append(f'{residualHistoryN}r')
    if config_parts:
        parts.append('(' + ','.join(config_parts) + ')')
    if accuracy is not None:
        parts.append(f'{stop_prefix}{accuracy*100:.2f}')
    elif stop_prefix:
        parts.append(stop_prefix)
    filename = '_'.join(parts) + '.pth'
    filepath = os.path.join(save_dir, filename)
    # 保存模型权重和归一化参数
    save_dict = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
    }
    if cfg is not None:
        save_dict['numAttentionHeads'] = cfg.get('numAttentionHeads', 1)
    torch.save(save_dict, filepath)
    print(f'模型已保存: {filepath}')
    if scaler is not None:
        print(f'归一化参数已保存')
#endregion

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
    data, train_mask, val_mask, test_mask, scaler = build_graph(stock_data_list, cfg)
    # 注入实际特征维度：Net中conv1输入维度和proj1历史注入维度据此动态确定，避免特征列数变化时维度硬编码失配。
    # 用新dict不污染原cfg（cfg会被打印到日志/超参数文件，保持干净）
    cfg = {**cfg, 'featDim': data.x.size(1)}
    model = Net(cfg).to(device)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    # 类别加权：用训练集统计各类别权重，平衡不平衡样本
    if cfg['ifOpenClassWeight']:
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
            print(f"conv1历史注入窗口: {cfg['residualHistoryN']}步拼接（维度 {model.feat_dim*cfg['residualHistoryN']}→32）")
        print(f"入边窗口: K={cfg['edgeWindowK']}, 稀疏间隔={cfg['edgeStride']}（每节点直接聚合前{cfg['edgeWindowK']}天内隔{cfg['edgeStride']}取一，边数={data.edge_index.shape[1]}）")

    precisions, recalls, f1s, losses = [], [], [], []
    # 初始化早停控制器
    early_stopper = EarlyStopper(cfg.get('earlyStopPatience', earlyStopPatience)) if ifOpenEarlyStop else None
    # 最佳F1初始化，用于记录训练过程中最佳验证F1及其出现轮次
    best_f1 = 0.0
    best_epoch = 0
    #模型训练/验证
    train_start = time.time()   #记录训练开始时间，用于统计耗时
    
    # 设置训练上下文（用于Ctrl+C中断时保存模型）
    global current_training_context
    current_training_context = {
        'model': model,
        'early_stopper': early_stopper,
        'cfg': cfg,
        'dataDate': dataDate,
        'stock_count': len(stock_data_list),
        'scaler': scaler,
        'data': data,
        'test_mask': test_mask,
        'periodRange': periodRange,
        'edgeWindowK': cfg['edgeWindowK'],
        'edgeStride': cfg['edgeStride'],
        'residualHistoryN': cfg['residualHistoryN']
    }
    
    # 进入模型训练模式（启用 Dropout 和 Batch Normalization 防止过拟合）
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, batch=data.batch)    #模型的输入有节点特征还有边特征,使用的是全部数据
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

    # 清除训练上下文（训练已完成）
    current_training_context = None

    #训练耗时统计
    train_elapsed = time.time() - train_start
    if not quiet:
        print(f"训练完成：最佳验证F1={best_f1:.4f}（第{best_epoch}轮），耗时 {int(train_elapsed//60)}分{train_elapsed%60:.0f}秒")

    #测试集评估
    metrics = evaluate_test(model, data, test_mask)
    return {'best_val_f1': best_f1, 'accuracy': metrics['accuracy'], 'precision': metrics['precision'],
            'recall': metrics['recall'], 'f1': metrics['f1'], 'cm': metrics['cm'],
            'model': model, 'best_epoch': best_epoch, 'elapsed': train_elapsed,
            'precisions': precisions, 'recalls': recalls, 'f1s': f1s, 'losses': losses,
            'scaler': scaler}

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


# ========== 单股票多模式对比运行函数（自包含，不影响现有__main__流程） ==========
def run_single_stock_compare(stockCode, modes, dataDate=dataDate, periodRange=periodRange, ifSaveModel=False):
    """
    单只股票多网络模式对比运行：数据采集→逐模式训练→对比表→热力图
    modes 传几个就跑几个，传 ['mixed'] 就是单模式，传 ['mixed','onlyGCN','onlyGAT'] 就是三模式对比

    :param stockCode: 股票代码，如 '000001.SZ'
    :param modes: 网络模式列表，如 ['mixed', 'onlyGCN', 'onlyGAT']
    :param dataDate: 数据截止日期
    :param periodRange: 向前取多少自然日
    :param ifSaveModel: 是否保存每个模式训练完的模型，默认False
    :return: list of (mode, result_dict, cfg)，失败返回 None
    """
    # --- 数据采集 ---
    if not useLocalData:
        bs.login()
    print(f'\n{"=" * 50}')
    print(f'开始处理股票: {stockCode}  (数据截止: {dataDate}, 周期: {periodRange}天)')
    print(f'运行模式: {modes}')
    print(f'{"=" * 50}')

    result = process_single_stock(stockCode, dataDate, periodRange)
    if not useLocalData:
        bs.logout()

    if result is None:
        msg = f'{stockCode} 数据不足或指标出错，跳过'
        print(msg)
        log_error(stockCode, msg)
        return None

    stock_data_list = [result]
    print(f'{stockCode} 预处理完成, 节点数: {len(result[0])}')

    # --- 构建基础配置（使用当前全局参数值） ---
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
        'numAttentionHeads': numAttentionHeads,
        'ifOpenGATConcat': ifOpenGATConcat,
    }

    # --- 逐模式训练 ---
    print(f'\n========== 网络模式对比：共 {len(modes)} 组 ==========')
    compare_results = []
    for mode in modes:
        print(f'\n----- [{mode}] -----')
        cfg = {**base_cfg, 'netMode': mode}
        r = run_training(cfg, stock_data_list, quiet=False)
        compare_results.append((mode, r, cfg))
        # 保存模型
        if ifSaveModel:
            save_trained_model(r['model'], dataDate, f'{stockCode}_{mode}', modelSaveDir, r.get('scaler'), accuracy=r['accuracy'], periodRange=periodRange, edgeWindowK=edgeWindowK, edgeStride=edgeStride, residualHistoryN=residualHistoryN, cfg=cfg)

    # --- 对比表 ---
    # 输出到控制台和txt文件（追加模式）
    log_comparison_result(stockCode, compare_results, f'对比结果_{dataDate}.txt')
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'对比结果_{dataDate}.txt')
    
    print(f'\n========== 对比结果 ({stockCode}) ==========')
    print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
    print('-' * 75)
    for mode, r, _ in compare_results:
        print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
    print('-' * 75)
    for mode, r, _ in compare_results:
        print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
    print(f'结果已追加到: {result_file}')
    return compare_results

#region 完整的消融实验训练模式，会生成对比图
def run_all_func(modes):
    """
    完整训练入口，按开关走三条主流程：消融实验 / 超参数搜索 / 单次训练
    :param modes: 网络模式列表，如 ['onlyGAT'] 只跑一种，或 ['mixed', 'onlyGCN', 'onlyGAT'] 跑多种对比；必填
    """
    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    # 遍历码表获取所有股票（每只股票内部按时序75/10/15划分train/val/test，后续拼成大图时各mask拼接）
    stock_data_list = []  # 每个元素: (priceDic, train_mask, val_mask, test_mask, code)

    if ifOpenMultiStock:
        # 多股票模式：遍历沪深300码表，每只股票独立处理后拼成大图
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        dataCount = 0
        for code in allStockSorted:
            current_code = code
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

    if not useLocalData:
        bs.logout()
    # 预处理结果检查：无可用数据时终止程序
    if len(stock_data_list) == 0:
        print('错误：没有成功预处理任何股票，程序终止')
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
        'numAttentionHeads': numAttentionHeads,
        'ifOpenGATConcat': ifOpenGATConcat,
    }
    if ifOpenAblation:
        # 消融实验模式：遍历各模式组训练，输出对比表+热力图，量化GCN/GAT对训练的影响（未传modes时用ablationModes）
        modeList = modes if modes else ablationModes
        print(f'========== 消融实验：网络结构模式对比，共{len(modeList)}组 ==========')
        ablation_results = []
        for mode in modeList:
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
            data_vis, _, _, _, _ = build_graph([vis_stock], cfg)
            input(f'\n按回车键查看 [{mode}] 各层热力图...')
            plot_attention_heatmaps(r['model'], data_vis, vis_stock[0], cfg, vis_stock[1], vis_stock[2], vis_stock[4], mode=mode)
        sys.exit(0)
    elif ifOpenHyperSearch:
        # 搜索模式下屏蔽sklearn的UndefinedMetricWarning（某类无预测样本时的警告），避免刷屏干扰每组摘要行；单次训练模式不屏蔽
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        # 固定采样种子，保证搜索组合可复现（需在run_training重置种子前一次性采样完，采样与训练互不影响）
        set_seed(2)
        trial_configs = sample_configs(hyperSearchSpace, hyperSearchTrials)
        modeList = modes
        totalTrials = len(trial_configs) * len(modeList)
        print(f'========== 超参数随机搜索：{len(trial_configs)}组参数 × {len(modeList)}种模式({modeList}) = 共{totalTrials}次训练，每次{hyperSearchTrainingTimes}轮 ==========')
        trial_results = []
        doneCount = 0
        for idx, cfg in enumerate(trial_configs):
            for mode in modeList:
                trial_cfg = {**cfg, 'netMode': mode}
                doneCount += 1
                r = run_training(trial_cfg, stock_data_list, quiet=True, epochs=hyperSearchTrainingTimes)
                trial_results.append((r['best_val_f1'], r, trial_cfg, doneCount))
                print(f"[trial {doneCount:2d}/{totalTrials}] valF1={r['best_val_f1']:.4f}(第{r['best_epoch']}轮) | test[Acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}] 耗时={r['elapsed']:.0f}s  {trial_cfg}")
        #按验证F1排序选最优（不看testF1，避免用测试集选模型造成评估泄露）
        trial_results.sort(key=lambda t: t[0], reverse=True)
        print('------ 搜索结果Top5（按验证F1排序） ------')
        for vf1, r, cfg, trial_no in trial_results[:5]:
            cm_str = str(r['cm']).replace('\n', ' ')
            print(f"[trial {trial_no:2d}/{totalTrials}] valF1={vf1:.4f}(第{r['best_epoch']}轮) | test[Acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}] 耗时={r['elapsed']:.0f}s {cm_str} {cfg}")
        best_cfg = trial_results[0][2]
        print(f'最佳配置: {best_cfg}')
        print('提示：将最佳配置手动填回参数区并关闭ifOpenHyperSearch，即可单次训练复现（种子固定，结果与搜索时一致，可看逐轮日志与训练曲线）')
        result = trial_results[0][1]  #直接使用搜索中最佳组的结果，不再重复精训
        result_cfg = best_cfg  #最佳组配置（已含netMode），供热力图重建图用
    else:
        #单次训练模式：按modes逐个训练
        modeList = modes
        single_results = []
        for mode in modeList:
            print(f'\n----- 单训练组 [{mode}] -----')
            cfg = {**base_cfg, 'netMode': mode}
            r = run_training(cfg, stock_data_list, quiet=False)
            single_results.append((mode, r, cfg))
        if len(single_results) > 1:
            #多模式时输出对比表（格式与消融实验一致）
            print('\n========== 单训练模式对比 ==========')
            print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
            print('-' * 75)
            for mode, r, _ in single_results:
                print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
            print('-' * 75)
            for mode, r, _ in single_results:
                print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
        #后续指标输出与热力图用验证F1最高的一组（单模式时即该组本身）
        best_mode, result, result_cfg = max(single_results, key=lambda t: t[1]['best_val_f1'])
        if len(single_results) > 1:
            print(f'后续指标输出与热力图采用验证F1最高的一组: [{best_mode}]')

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
        used_cfg = best_cfg if ifOpenHyperSearch else result_cfg
        if used_cfg['edgeWindowK'] <= 1:
            print('注意力热力图跳过：edgeWindowK<=1时每节点仅1条入边，softmax后注意力恒为1，无展示意义')
        else:
            vis_stock = stock_data_list[0]  # (priceDic, train_mask, val_mask, test_mask, code)
            data_vis, _, _, _, _ = build_graph([vis_stock], used_cfg)
            input('\n按回车键查看GAT注意力热力图...')
            plot_attention_heatmaps(result['model'], data_vis, vis_stock[0], used_cfg, vis_stock[1], vis_stock[2], vis_stock[4])

    # 训练过程参数变化可视化（按回车后显示图表）
    #input('\n按回车键查看训练指标曲线图...')
    #plot_metrics(result['precisions'], result['recalls'], result['f1s'], result['losses'])
#endregion

#region 轻量版训练入口：只输出指标和混淆矩阵，不画任何图
def run_all_func_lite(modes):
    """
    轻量版完整训练入口，按开关走三条主流程：消融实验 / 超参数搜索 / 单次训练
    与run_all_func的区别：不画热力图、不弹图表、不需要按回车，只输出文本指标
    :param modes: 网络模式列表，如 ['onlyGAT'] 只跑一种，或 ['mixed', 'onlyGCN', 'onlyGAT'] 跑多种对比；必填
    日志：运行期间本函数输出到控制台的所有信息，经log_print原样追加写入 超参数_{dataDate}_{maxStockCount}s_({periodRange}d,{len(modes)}m,head,{hyperSearchTrials}round).txt
    """
    global current_code, allStockSorted, _active_log_file
    # 打开日志文件：本函数控制台输出内容原样写入txt（追加模式，每行写入后立即flush防中断丢失）
    _active_log_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'超参数_{dataDate}_{maxStockCount}s_({periodRange}d,{len(modes)}m,head,{hyperSearchTrials}round).txt'), 'a', encoding='utf-8')
    if not useLocalData:
        bs.login()
    stock_data_list = []

    if ifOpenMultiStock:
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        dataCount = 0
        for code in allStockSorted:
            current_code = code
            if len(stockPoolList) == 0 or code not in stockPoolList:
                continue
            if maxStockCount is not None and dataCount >= maxStockCount:
                break
            try:
                result = process_single_stock(code, dataDate, periodRange)
                if result is not None:
                    stock_data_list.append(result)
                    dataCount += 1
                    log_print(f'{code} 预处理完成,序号:NO.{dataCount},节点数:{len(result[0])}')
                else:
                    log_print(code + ' 数据不足或指标出错,跳过')
            except Exception as ex:
                log_print("失败代码："+code+"，异常信息："+str(ex))
        log_print(f'共预处理 {len(stock_data_list)} 只股票，总节点数 {sum(len(r[0]) for r in stock_data_list)}')
    else:
        result = process_single_stock(stockCode, dataDate, periodRange)
        if result is not None:
            stock_data_list.append(result)
            log_print(f'{stockCode} 预处理完成,节点数:{len(result[0])}')
        else:
            log_print(f'{stockCode} 数据不足或指标出错')

    if not useLocalData:
        bs.logout()
    if len(stock_data_list) == 0:
        log_print('错误：没有成功预处理任何股票，程序终止')
        sys.exit(1)

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
        'numAttentionHeads': numAttentionHeads,
        'ifOpenGATConcat': ifOpenGATConcat,
    }

    if ifOpenAblation:
        # 消融实验模式
        modeList = modes if modes else ablationModes
        log_print(f'\n========== 消融实验：网络结构模式对比，共{len(modeList)}组 ==========')
        ablation_results = []
        for mode in modeList:
            log_print(f'\n----- 消融组 [{mode}] -----')
            cfg = {**base_cfg, 'netMode': mode}
            r = run_training(cfg, stock_data_list, quiet=False)
            ablation_results.append((mode, r, cfg))
        # 对比表（经log_print同步写入txt）
        log_print('\n========== 消融实验结果对比 ==========')
        log_print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
        log_print('-' * 75)
        for mode, r, _ in ablation_results:
            log_print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
        log_print('-' * 75)
        for mode, r, _ in ablation_results:
            log_print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
        _active_log_file.close()
        _active_log_file = None
        return ablation_results

    elif ifOpenHyperSearch:
        # 超参数搜索模式
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        set_seed(2)
        trial_configs = sample_configs(hyperSearchSpace, hyperSearchTrials)
        modeList = modes
        totalTrials = len(trial_configs) * len(modeList)
        log_print(f'\n========== 超参数随机搜索：{len(trial_configs)}组参数 × {len(modeList)}种模式({modeList}) = 共{totalTrials}次训练，每次{hyperSearchTrainingTimes}轮 ==========')
        trial_results = []
        doneCount = 0
        for idx, cfg in enumerate(trial_configs):
            for mode in modeList:
                trial_cfg = {**cfg, 'netMode': mode}
                doneCount += 1
                try:
                    r = run_training(trial_cfg, stock_data_list, quiet=True, epochs=hyperSearchTrainingTimes)
                except Exception as ex:
                    log_print(f"[trial {doneCount:2d}/{totalTrials}] 训练失败，跳过：{ex}  {trial_cfg}")
                    continue
                trial_results.append((r['best_val_f1'], r, trial_cfg, doneCount))
                # 单行紧凑格式，控制台与txt经log_print完全一致输出（混淆矩阵换行替换为空格，保证一行一条）
                cm_str = str(r['cm']).replace('\n', ' ')
                log_print(f"[trial {doneCount:2d}/{totalTrials}] valF1={r['best_val_f1']:.4f}(第{r['best_epoch']}轮) | test[Acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}] 耗时={r['elapsed']:.0f}s {cm_str} {trial_cfg}")
        trial_results.sort(key=lambda t: t[0], reverse=True)
        if len(trial_results) == 0:
            log_print('错误：所有trial均训练失败，无搜索结果')
            _active_log_file.close()
            _active_log_file = None
            return None
        log_print('\n------ 搜索结果Top5（按验证F1排序） ------')
        for vf1, r, cfg, trial_no in trial_results[:5]:
            cm_str = str(r['cm']).replace('\n', ' ')
            log_print(f"[trial {trial_no:2d}/{totalTrials}] valF1={vf1:.4f}(第{r['best_epoch']}轮) | test[Acc={r['accuracy']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}] 耗时={r['elapsed']:.0f}s {cm_str} {cfg}")
        best_cfg = trial_results[0][2]
        log_print(f'\n最佳配置: {best_cfg}')
        log_print('提示：将最佳配置手动填回参数区并关闭ifOpenHyperSearch，即可单次训练复现（种子固定，结果与搜索时一致，可看逐轮日志与训练曲线）')
        result = trial_results[0][1]
        # 输出最佳组的测试集结果（经log_print同步写入txt）
        log_print('\n==============================')
        log_print('测试集评估结果')
        log_print('==============================')
        log_print('Accuracy:  {:.2f}%'.format(result['accuracy'] * 100))
        log_print('Precision: {:.4f}'.format(result['precision']))
        log_print('Recall:    {:.4f}'.format(result['recall']))
        log_print('F1 (macro): {:.4f}'.format(result['f1']))
        log_print('------------------------------')
        log_print('混淆矩阵 (行=真实, 列=预测):')
        log_print(result['cm'])
        log_print('==============================')
        _active_log_file.close()
        _active_log_file = None
        return result

    else:
        # 单次训练模式
        modeList = modes
        single_results = []
        for mode in modeList:
            log_print(f'\n----- 单训练组 [{mode}] -----')
            cfg = {**base_cfg, 'netMode': mode}
            r = run_training(cfg, stock_data_list, quiet=False)
            single_results.append((mode, r, cfg))
        if len(single_results) > 1:
            log_print('\n========== 单训练模式对比 ==========')
            log_print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
            log_print('-' * 75)
            for mode, r, _ in single_results:
                log_print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
            log_print('-' * 75)
            for mode, r, _ in single_results:
                log_print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
            best_mode, result, result_cfg = max(single_results, key=lambda t: t[1]['best_val_f1'])
            log_print(f'验证F1最高的一组: [{best_mode}]')
        else:
            mode, result, cfg = single_results[0]
        # 输出测试集结果（经log_print同步写入txt）
        log_print('\n==============================')
        log_print('测试集评估结果')
        log_print('==============================')
        log_print('Accuracy:  {:.2f}%'.format(result['accuracy'] * 100))
        log_print('Precision: {:.4f}'.format(result['precision']))
        log_print('Recall:    {:.4f}'.format(result['recall']))
        log_print('F1 (macro): {:.4f}'.format(result['f1']))
        log_print('------------------------------')
        log_print('混淆矩阵 (行=真实, 列=预测):')
        log_print(result['cm'])
        log_print('==============================')
        _active_log_file.close()
        _active_log_file = None
        return result
#endregion

#region ========== 方式二：单股票遍历训练（封装函数） ==========
def run_method_two(stock_list=None, modes=None, resume_from='', ifSaveModel=False):
    """
    方式二：单股票遍历训练，每只股票独立训练一个模型，支持对比多种网络模式
    :param stock_list: 手动指定股票列表，如 ['000009.SZ', '000010.SZ']；None或空则自动从沪深300码表获取
    :param modes: 网络模式列表，如 ['onlyGAT'] 单模式 或 ['mixed', 'onlyGAT'] 对比模式；必须有值
    :param resume_from: 断点续跑，填入股票代码（如'000023.SZ'），只跑该代码及之后的股票；空则从头开始
    :param ifSaveModel: 是否保存每只股票训练完的模型，默认False
    """
    if stock_list is None:
        stock_list = []
    if not modes:
        print('错误：modes 必须有值，如 ["onlyGAT"] 或 ["mixed", "onlyGAT"]')
        sys.exit(1)

    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    stockPoolList = []
    if stock_list:
        allStockSorted = stock_list
        print(f'\n========== 方式二-手动模式：共 {len(stock_list)} 只股票 ==========')
    else:
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        if resume_from:
            filtered = []
            for code in allStockSorted:
                if code >= resume_from:
                    filtered.append(code)
            allStockSorted = filtered
            print(f'\n========== 方式二-断点续跑：从 {resume_from} 开始，剩余 {len(allStockSorted)} 只 ==========')
        print(f'\n========== 方式二-自动模式：码表 {len(stockPoolList)} 只，全量 {len(allStockDict)} 只，本次遍历 {len(allStockSorted)} 只 ==========')

    if not useLocalData:
        bs.logout()
    failed_list = []
    for code in allStockSorted:
        current_code = code
        if not stock_list and (len(stockPoolList) == 0 or code not in stockPoolList):
            continue
        try:
            result = run_single_stock_compare(code, modes, ifSaveModel=ifSaveModel)
        except Exception as e:
            print(f'{code} 运行失败，跳过: {e}')
            log_error(code, traceback.format_exc())
            failed_list.append(code)

    if failed_list:
        log_error('首轮失败列表', f'共{len(failed_list)}只: {failed_list}')
        retry_round = 1
        sleep_seconds = 60
        while failed_list:
            prev_count = len(failed_list)
            print(f'\n========== 第{retry_round}轮重试：{len(failed_list)} 只失败股票，休眠{sleep_seconds}秒后开始 ==========')
            time.sleep(sleep_seconds)
            retry_list = failed_list
            failed_list = []
            for code in retry_list:
                current_code = code
                try:
                    result = run_single_stock_compare(code, modes, ifSaveModel=ifSaveModel)
                except Exception as e:
                    print(f'{code} 重试仍失败，跳过: {e}')
                    log_error(code, traceback.format_exc())
                    failed_list.append(code)
            if failed_list:
                log_error(f'第{retry_round}轮重试失败列表', f'共{len(failed_list)}只: {failed_list}')
                if len(failed_list) >= prev_count:
                    sleep_seconds *= 2
                    print(f'本轮无改善，下次休眠时间调整为{sleep_seconds}秒')
                else:
                    sleep_seconds = 60
            retry_round += 1
    print('方式二运行完成')
#endregion

#region ========== 方式三：多股票拼大图训练（封装函数） ==========
def run_method_three(stock_list_multi=None, compare_modes_multi=None, ifSaveModel=False):
    """
    方式三：多股票拼大图训练，支持单模式或对比模式，训练完成后自动保存模型
    :param stock_list_multi: 手动指定股票列表，如 ['000009.SZ', '000010.SZ']；None或空则自动从码表获取，最多取 maxStockCount 只
    :param compare_modes_multi: 网络模式列表，如 ['onlyGAT'] 单模式 或 ['mixed', 'onlyGAT'] 对比模式；必须有值
    :param ifSaveModel: 训练完成后是否保存模型，默认False
    """
    if stock_list_multi is None:
        stock_list_multi = []
    if not compare_modes_multi:
        print('错误：compare_modes_multi 必须有值，如 ["onlyGAT"] 或 ["mixed", "onlyGAT"]')
        sys.exit(1)

    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    stockPoolList_multi = []
    if stock_list_multi:
        allStockSorted_multi = stock_list_multi
        print(f'\n========== 方式三-手动模式：共 {len(stock_list_multi)} 只股票 ==========')
    else:
        stockPoolList_multi = get_stock_pool_list('hs300')
        allStockDict_multi = get_stock_pool_list('all')
        allStockSorted_multi = sorted(allStockDict_multi.keys())
        print(f'\n========== 方式三-自动模式：码表 {len(stockPoolList_multi)} 只，全量 {len(allStockDict_multi)} 只，本次遍历 {len(allStockSorted_multi)} 只 ==========')

    allStockSorted = allStockSorted_multi
    stock_data_list_multi = []
    dataCount_multi = 0
    for code in allStockSorted_multi:
        current_code = code
        if not stock_list_multi and (len(stockPoolList_multi) == 0 or code not in stockPoolList_multi):
            continue
        if maxStockCount is not None and dataCount_multi >= maxStockCount:
            print(f'已达到最大股票数 {maxStockCount}，停止预处理')
            break
        try:
            result = process_single_stock(code, dataDate, periodRange)
            if result is not None:
                stock_data_list_multi.append(result)
                dataCount_multi += 1
                print(f'{code} 预处理完成, 序号: NO.{dataCount_multi}, 节点数: {len(result[0])}')
            else:
                print(f'{code} 数据不足或指标出错，跳过')
                log_error(code, f'{code} 数据不足或指标出错')
        except Exception as e:
            print(f'{code} 预处理失败，跳过: {e}')
            log_error(code, traceback.format_exc())

    print(f'\n共预处理 {len(stock_data_list_multi)} 只股票，总节点数 {sum(len(r[0]) for r in stock_data_list_multi)}')

    if not useLocalData:
        bs.logout()
    if len(stock_data_list_multi) == 0:
        print('错误：没有成功预处理任何股票，程序终止')
        sys.exit(1)

    base_cfg_multi = {
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
        'numAttentionHeads': numAttentionHeads,
        'ifOpenGATConcat': ifOpenGATConcat,
    }

    if len(compare_modes_multi) > 1:
        # 对比模式：遍历多个网络结构
        print(f'\n========== 方式三-对比模式：{len(compare_modes_multi)} 种网络结构对比 ==========')
        comparison_results_multi = []
        for mode in compare_modes_multi:
            print(f'\n----- 训练模式 [{mode}] -----')
            cfg = {**base_cfg_multi, 'netMode': mode}
            result_multi = run_training(cfg, stock_data_list_multi, quiet=False)
            comparison_results_multi.append((mode, result_multi, cfg))
            if ifSaveModel:
                save_trained_model(result_multi['model'], dataDate, mode, modelSaveDir, result_multi.get('scaler'), len(stock_data_list_multi), result_multi['accuracy'], periodRange=periodRange, edgeWindowK=edgeWindowK, edgeStride=edgeStride, residualHistoryN=residualHistoryN, cfg=cfg)
        print(f'\n========== 对比结果 (方式三-多股票拼大图) ==========')
        print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
        print('-' * 75)
        for mode, r, _ in comparison_results_multi:
            print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
        print('-' * 75)
        for mode, r, _ in comparison_results_multi:
            print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
        log_comparison_result('方式三-多股票拼大图', comparison_results_multi, f'对比结果_{dataDate}.txt')
    else:
        # 单模式：用 compare_modes_multi[0]
        mode = compare_modes_multi[0]
        print(f'\n========== 方式三：开始训练（{len(stock_data_list_multi)} 只股票拼成大图，模式: {mode}） ==========')
        cfg = {**base_cfg_multi, 'netMode': mode}
        result_multi = run_training(cfg, stock_data_list_multi, quiet=False)
        if ifSaveModel:
            save_trained_model(result_multi['model'], dataDate, mode, modelSaveDir, result_multi.get('scaler'), len(stock_data_list_multi), result_multi['accuracy'], periodRange=periodRange, edgeWindowK=edgeWindowK, edgeStride=edgeStride, residualHistoryN=residualHistoryN, cfg=cfg)
        print('\n==============================')
        print(f'方式三-测试集评估结果（模式: {mode}）')
        print('==============================')
        print('Accuracy:  {:.2f}%'.format(result_multi['accuracy'] * 100))
        print('Precision: {:.4f}'.format(result_multi['precision']))
        print('Recall:    {:.4f}'.format(result_multi['recall']))
        print('F1 (macro): {:.4f}'.format(result_multi['f1']))
        print('------------------------------')
        print('混淆矩阵 (行=真实, 列=预测):')
        print(result_multi['cm'])
        print('==============================')
        # 构造对比结果列表（与对比模式格式一致）
        single_results = [(mode, result_multi, cfg)]
        log_comparison_result('方式三-多股票拼大图', single_results, f'对比结果_{dataDate}.txt')
#endregion

#region ========== 方式四：加载已保存模型直接预测（封装函数） ==========
def run_method_four(model_name, net_mode, stock_list=None):
    """
    方式四：加载已保存的模型直接预测，不训练
    :param model_name: 模型名称（不含.pth后缀，如 '20250101_mixed'、'20250101_onlyGAT'）
    :param net_mode: 该模型训练时的网络模式（必须与训练时一致：mixed/onlyGCN/onlyGAT）
    :param stock_list: 手动指定股票列表，如 ['000009.SZ', '000010.SZ']；None或空则自动从沪深300获取，最多取 maxStockCount 只
    """
    if stock_list is None:
        stock_list = []
    if not model_name:
        print('错误：model_name 必须有值，如 "20250101_onlyGAT"')
        sys.exit(1)
    if not net_mode:
        print('错误：net_mode 必须有值，如 "onlyGAT"')
        sys.exit(1)

    filepath = os.path.join(modelSaveDir, f'{model_name}.pth')
    if not os.path.exists(filepath):
        print(f'\n错误：模型文件不存在 {filepath}')
        print(f'当前可用模型：')
        if os.path.exists(modelSaveDir):
            for f in sorted(os.listdir(modelSaveDir)):
                if f.endswith('.pth'):
                    print(f'  {f.replace(".pth", "")}')
        else:
            print(f'  保存目录不存在: {modelSaveDir}')
        return

    print(f'\n========== 方式四：加载模型 {model_name} 直接预测 ==========')
    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    stockPoolList = []
    if stock_list:
        allStockSorted = stock_list
        print(f'手动模式：共 {len(stock_list)} 只股票')
    else:
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        print(f'自动模式：码表 {len(stockPoolList)} 只，全量 {len(allStockDict)} 只，本次遍历 {len(allStockSorted)} 只')

    stock_data_list = []
    dataCount = 0
    for code in allStockSorted:
        current_code = code
        if not stock_list and (len(stockPoolList) == 0 or code not in stockPoolList):
            continue
        if maxStockCount is not None and dataCount >= maxStockCount:
            print(f'已达到最大股票数 {maxStockCount}，停止预处理')
            break
        try:
            result = process_single_stock(code, dataDate, periodRange)
            if result is not None:
                stock_data_list.append(result)
                dataCount += 1
                print(f'{code} 预处理完成, 序号: NO.{dataCount}, 节点数: {len(result[0])}')
            else:
                print(f'{code} 数据不足或指标出错，跳过')
        except Exception as e:
            print(f'{code} 预处理失败，跳过: {e}')
            log_error(code, traceback.format_exc())

    print(f'\n共预处理 {len(stock_data_list)} 只股票，总节点数 {sum(len(r[0]) for r in stock_data_list)}')

    if not useLocalData:
        bs.logout()
    if len(stock_data_list) == 0:
        print('错误：没有成功预处理任何股票，程序终止')
        sys.exit(1)

    # 先加载checkpoint：从模型文件读取numAttentionHeads，确保Net结构与训练时一致
    checkpoint = torch.load(filepath, weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    scaler = checkpoint.get('scaler')
    cfg = {
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
        'numAttentionHeads': checkpoint.get('numAttentionHeads', 1),
        'netMode': net_mode,
    }
    if scaler is not None:
        print(f'已加载归一化参数')
    else:
        print(f'警告：模型未包含归一化参数')
    
    data, _, _, test_mask, _ = build_graph(stock_data_list, cfg, scaler)
    model = Net(cfg).to(device)
    model_state_dict = convert_gat_state_dict(model_state_dict, model.state_dict())
    model.load_state_dict(model_state_dict)
    print(f'已加载模型: {filepath}')

    metrics = evaluate_test(model, data, test_mask)
    metrics['best_val_f1'] = 0.0
    metrics['best_epoch'] = 0
    metrics['elapsed'] = 0.0
    results = [(net_mode, metrics, cfg)]

    print(f'\n{"=" * 60}')
    print(f'方式四-{model_name}-{len(stock_data_list)}只  运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 60}')
    print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
    print('-' * 75)
    for mode, r, _ in results:
        print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
    print('-' * 75)
    for mode, r, _ in results:
        print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')
    log_comparison_result(f'方式四-{model_name}-{len(stock_data_list)}只', results, f'对比结果_{dataDate}.txt')
#endregion

#region ========== 方式五：滚动预测（模拟真实交易场景） ==========
def run_method_four_rolling(model_name, net_mode, stock_list=None, test_ratio=0.15):
    """
    方式五：滚动预测，模拟真实交易场景
    对每只股票的测试集，逐天进行预测，每天只用截止到当天的数据
    :param model_name: 模型名称（不含.pth后缀）
    :param net_mode: 网络模式（必须与训练时一致）
    :param stock_list: 股票列表，None或空则自动从沪深300获取
    :param test_ratio: 测试集比例，默认0.15
    """
    if stock_list is None:
        stock_list = []
    if not model_name:
        print('错误：model_name 必须有值')
        sys.exit(1)
    if not net_mode:
        print('错误：net_mode 必须有值')
        sys.exit(1)

    filepath = os.path.join(modelSaveDir, f'{model_name}.pth')
    if not os.path.exists(filepath):
        print(f'\n错误：模型文件不存在 {filepath}')
        return

    print(f'\n========== 方式五-滚动预测：加载模型 {model_name} ==========')
    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    stockPoolList = []
    if stock_list:
        allStockSorted = stock_list
        print(f'手动模式：共 {len(stock_list)} 只股票')
    else:
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        print(f'自动模式：码表 {len(stockPoolList)} 只，全量 {len(allStockDict)} 只')

    # 归一化处理：保持和原始训练一致，用前75%的数据计算归一化参数
    # 先加载checkpoint：从模型文件读取numAttentionHeads，确保Net结构与训练时一致
    checkpoint = torch.load(filepath, weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    scaler = checkpoint.get('scaler')
    cfg = {
        'ifOpenNormalize': ifOpenNormalize,  # 使用和训练时相同的归一化设置
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
        'numAttentionHeads': checkpoint.get('numAttentionHeads', 1),
        'netMode': net_mode,
    }

    # 加载模型和归一化参数
    if scaler is not None:
        print(f'已加载归一化参数（训练时保存的）')
    else:
        print(f'警告：模型未包含归一化参数')
        
    model = Net(cfg).to(device)
    model_state_dict = convert_gat_state_dict(model_state_dict, model.state_dict())
    model.load_state_dict(model_state_dict)
    print(f'已加载模型: {filepath}')

    all_predictions = []  # 存储所有预测结果
    all_labels = []       # 存储所有真实标签
        
    for code in allStockSorted:
        current_code = code
        if not stock_list and (len(stockPoolList) == 0 or code not in stockPoolList):
            continue
            
        try:
            # 获取完整数据
            result = process_single_stock(code, dataDate, periodRange)
            if result is None:
                print(f'{code} 数据不足，跳过')
                continue
                
            priceDic, _, _, _, _ = result
            items = list(priceDic.items())
            total_days = len(items)
            test_start = int(total_days * (1 - test_ratio))
            test_days = total_days - test_start
                
            print(f'\n{code}: 共{total_days}天，测试集从第{test_start+1}天开始，共{test_days}天')
                
            # 滚动预测：从测试集第一天开始
            for test_day_idx in range(test_days):
                test_day = test_start + test_day_idx + 1  # 当前要预测的天数（1-based）
                    
                # 截取数据：只用前 test_day-1 天建图，末节点即"预测第test_day天flag"的读出节点
                # 标签前移一天后，节点i的输出对应第i+1天的flag，因此不需再补零节点
                graph_days = test_day - 1
                if graph_days < 2:
                    continue
                truncated_priceDic = dict(items[:graph_days])
                    
                # 构建单只股票的stock_data_list
                # 关键：保持和原始训练相同的划分比例（75%/10%/15%）
                # 归一化只用前75%的数据计算，保证一致性
                split_train = int(graph_days * 0.75)
                split_val = int(graph_days * 0.85)
                    
                train_mask = [i < split_train for i in range(graph_days)]      # 前75%为True
                val_mask = [split_train <= i < split_val for i in range(graph_days)]  # 75%-85%为True
                test_mask = [i == graph_days - 1 for i in range(graph_days)]   # 只有最后一个节点为True
                    
                single_stock_data = [(truncated_priceDic, train_mask, val_mask, test_mask, code)]
                    
                # 建图（使用训练时的归一化参数）
                data, _, _, test_mask_out, _ = build_graph(single_stock_data, cfg, scaler)
                    
                # 读出节点=最后一个节点，它的输出即对第test_day天flag的预测
                last_node_idx = data.x.size(0) - 1
                    
                # 预测
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index, batch=data.batch)
                    pred = torch.argmax(out[last_node_idx]).item()
                # 真实标签取第test_day天的flag（截断图末节点的y是-1，不可用）
                label = items[test_day - 1][1]['flag']
                    
                all_predictions.append(pred)
                all_labels.append(label)
                    
                # 每10天输出一次进度
                if (test_day_idx + 1) % 10 == 0 or test_day_idx == 0:
                    print(f'  第{test_day}天预测: 预测={pred}, 真实={label} {"✓" if pred == label else "✗"}')
        except Exception as e:
            print(f'{code} 处理失败: {e}')
            traceback.print_exc()
            continue
    
    if not useLocalData:
        bs.logout()
    # 计算整体指标
    if len(all_predictions) == 0:
        print('\n没有成功的预测')
        return
    
    
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 记录结果
    results = [(net_mode, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'best_val_f1': 0.0,
        'best_epoch': 0,
        'elapsed': 0.0
    }, cfg)]

    print(f'\n{"=" * 60}')
    print(f'方式五-{model_name}-{len(all_predictions)}次  运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 60}')
    print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
    print('-' * 75)
    for mode, r, _ in results:
        print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
    print('-' * 75)
    for mode, r, _ in results:
        print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')

    log_comparison_result(f'方式五-{model_name}-{len(all_predictions)}次', results, f'对比结果_{dataDate}.txt')
#endregion

#region ========== 方式六：实盘/回测预测（可切换模式） ==========
def run_live_predict(model_name, net_mode, stock_list=None, is_live=True, max_count=None):
    """
    实盘/回测预测（可切换模式）
    :param model_name: 模型名称（不含.pth后缀）
    :param net_mode: 网络模式（必须与训练时一致：mixed/onlyGCN/onlyGAT）
    :param stock_list: 要预测的股票列表；None或空则自动从沪深300获取
    :param is_live: True=预测未来（明天），保存结果到txt；False=回测历史（最后一天），评估准确率
    :param max_count: 自动模式下最多预测多少只股票（None=不限制，自动模式生效）
    """
    if not model_name or not net_mode:
        print('错误：model_name、net_mode 必须有值')
        return

    filepath = os.path.join(modelSaveDir, f'{model_name}.pth')
    if not os.path.exists(filepath):
        print(f'错误：模型文件不存在 {filepath}')
        return

    mode_str = '实盘预测（明天）' if is_live else '回测验证（最后一天）'
    print(f'\n========== 方式六-{mode_str}：模型={model_name}，数据截止={dataDate} ==========')

    if stock_list is None:
        stock_list = []
    global current_code, allStockSorted
    if not useLocalData:
        bs.login()
    stockPoolList = []
    # 获取股票列表
    if stock_list:
        allStockSorted = stock_list
        print(f'手动模式：共 {len(stock_list)} 只股票')
    else:
        stockPoolList = get_stock_pool_list('hs300')
        allStockDict = get_stock_pool_list('all')
        allStockSorted = sorted(allStockDict.keys())
        print(f'自动模式：码表 {len(stockPoolList)} 只，全量 {len(allStockDict)} 只')

    # 本地模式：预检数据覆盖（实盘和回测均要求数据覆盖dataDate）
    if useLocalData:
        check_codes = list(stock_list) if stock_list else list(stockPoolList)
        if not check_codes:
            check_codes = allStockSorted[:30]  # 兜底：拿不到股票池时抽查码表前30只
        ok, check_msg = check_local_data_coverage(check_codes, dataDate)
        if not ok:
            print(check_msg)
            return

    # 加载模型
    # 先加载checkpoint：从模型文件读取numAttentionHeads，确保Net结构与训练时一致
    checkpoint = torch.load(filepath, weights_only=False)
    cfg = {
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
        'numAttentionHeads': checkpoint.get('numAttentionHeads', 1),
        'netMode': net_mode,
    }
    model = Net(cfg).to(device)
    # 兼容不同PyG版本的GATConv参数命名
    state_dict = convert_gat_state_dict(checkpoint['model_state_dict'], model.state_dict())
    model.load_state_dict(state_dict)
    scaler = checkpoint.get('scaler')
    model.eval()
    print(f'已加载模型: {filepath}')
    if scaler is not None:
        print('已加载归一化参数')

    # 回测模式：用于评估
    all_predictions = []
    all_labels = []
    all_codes = []
    # 实盘模式：用于保存结果
    live_results = []
    # 计数
    predict_count = 0

    print(f'\n{"股票代码":<12}{"预测日期":<12}{"预测结果":>8}{"含义":>10}')
    print('-' * 50)
    for code in allStockSorted:
        current_code = code
        if not stock_list and (len(stockPoolList) == 0 or code not in stockPoolList):
            continue
        if max_count is not None and predict_count >= max_count:
            print(f'已达到最大股票数 {max_count}，停止预测')
            break
        try:
            # 获取截止到 dataDate 的全部数据
            result = process_single_stock(code, dataDate, periodRange)
            if result is None:
                print(f'{code:<12}{"跳过":<12}{"--":>8}{"数据不足":>10}')
                continue

            priceDic, _, _, _, _ = result
            items = list(priceDic.items())
            total_days = len(items)
            last_trade_date = items[-1][0]  # 最后一个交易日

            if is_live:
                # 实盘模式：用全部N天建图，末节点（第N天）的输出即对下一个交易日的预测
                graph_days = total_days  # 用全部数据建图
                predict_label = None     # 明天未知，无法评估
                predict_date = f'{dataDate}next'  # 表示预测的是下一个交易日
            else:
                # 回测模式：用前N-1天建图，末节点（第N-1天）的输出即对第N天（最后一天）的预测
                graph_days = total_days - 1  # 排除最后一天
                last_day_data = items[-1][1]
                predict_label = last_day_data.get('flag', 0)  # 真实标签
                predict_date = last_trade_date

            # 建图
            truncated_priceDic = dict(items[:graph_days])
            train_mask = [True] * graph_days
            val_mask = [False] * graph_days
            test_mask = [False] * graph_days
            stock_data = [(truncated_priceDic, train_mask, val_mask, test_mask, code)]
            data, _, _, _, _ = build_graph(stock_data, cfg, scaler)

            # 读出节点=图中最后一个真实节点
            # 标签前移一天后，节点i的输出对应第i+1天的flag，因此不需再补零节点和补边
            predict_idx = data.x.size(0) - 1

            # 预测
            with torch.no_grad():
                out = model(data.x, data.edge_index, batch=data.batch)
                pred = torch.argmax(out[predict_idx]).item()

            flag_meaning = '买入/持有' if pred == 1 else '卖出/观望'
            print(f'{code:<12}{predict_date:<12}{pred:>8}{flag_meaning:>10}  （建图天数: {graph_days}）')

            # 收集结果
            if is_live:
                live_results.append((code, predict_date, pred))
            else:
                all_predictions.append(pred)
                all_labels.append(predict_label)
                all_codes.append(code)
            predict_count += 1

        except Exception as e:
            print(f'{code:<12}{"失败":<12}{"--":>8}{str(e)[:10]:>10}')
            traceback.print_exc()
            continue

    if not useLocalData:
        bs.logout()
    # 输出结果
    if is_live:
        # 实盘模式：保存到文件
        if len(live_results) > 0:
            filename = f'实盘预测_{dataDate.replace("-", "")}.txt'
            filepath = os.path.join(modelSaveDir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('股票代码,预测日期,预测flag\n')
                for code, date, flag in live_results:
                    f.write(f'{code},{date},{flag}\n')
            print(f'\n已保存预测结果到: {filepath}')
            print(f'共预测 {len(live_results)} 只股票')
    else:
        # 回测模式：计算评估指标
        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            cm = confusion_matrix(all_labels, all_predictions)

            # 记录结果
            results = [(net_mode, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cm': cm,
                'best_val_f1': 0.0,
                'best_epoch': 0,
                'elapsed': 0.0
            }, cfg)]

            print(f'\n{"=" * 60}')
            print(f'方式六-回测-{model_name}-{len(all_predictions)}只  运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'{"=" * 60}')
            print(f'{"模式":<10}{"Acc":>9}{"Precision":>11}{"Recall":>9}{"F1":>9}{"valF1":>9}{"轮次":>7}{"耗时(s)":>10}')
            print('-' * 75)
            for mode, r, _ in results:
                print(f'{mode:<10}{r["accuracy"]*100:>8.2f}%{r["precision"]:>11.4f}{r["recall"]:>9.4f}{r["f1"]:>9.4f}{r["best_val_f1"]:>9.4f}{r["best_epoch"]:>7d}{r["elapsed"]:>10.0f}')
            print('-' * 75)
            for mode, r, _ in results:
                print(f'[{mode}] 混淆矩阵 (行=真实, 列=预测):\n{r["cm"]}\n')

            # 输出每只股票的预测结果
            print(f'\n详细结果:')
            print(f'{"股票代码":<12}{"预测":>6}{"真实":>6}{"结果":>6}')
            print('-' * 35)
            for code, pred, label in zip(all_codes, all_predictions, all_labels):
                result_str = '✓' if pred == label else '✗'
                print(f'{code:<12}{pred:>6}{label:>6}{result_str:>6}')

            log_comparison_result(f'方式六-回测-{model_name}-{len(all_predictions)}只', results, f'对比结果_{dataDate}.txt')
#endregion

if __name__ == '__main__':
    #region 注册信号处理器（Ctrl+C 和 kill 命令）
    import signal
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 命令
    #endregion

    #run_all_func(modes=['onlyGAT'])
    run_all_func_lite(modes=['mixed', 'onlyGAT'])

    #region 方式一：手动指定股票列表
    #stock_list = ['000009.SZ', '000010.SZ', '000011.SZ', '000012.SZ', '000013.SZ', '000014.SZ', '000015.SZ', '000016.SZ', '000017.SZ']
    #for code in stock_list:
    #    try:
    #        run_single_stock_compare(code, ['mixed', 'onlyGAT'])
    #    except Exception as e:
    #        print(f'{code} 运行失败，跳过: {e}')
    #        log_error(code, traceback.format_exc())
    #        continue
    #endregion
    

    # ====== 方式二：单股票遍历训练（每只股票独立训练一个模型） ======
    #run_method_two(modes=['onlyGAT'], ifSaveModel=True)  # 单模式训练，自动从沪深300遍历所有股票，保存模型
    #run_method_two(modes=['mixed', 'onlyGAT'], ifSaveModel=True)  # 对比多种网络模式，自动从沪深300遍历，保存模型
    #run_method_two(stock_list=['000001.SZ'], modes=['mixed', 'onlyGCN', 'onlyGAT'], ifSaveModel=True)  # 手动指定股票列表，保存模型
    #run_method_two(modes=['onlyGAT'], resume_from='000023.SZ')  # 断点续跑，遍历沪深300股票池，从指定股票代码开始继续训练（不保存模型）

    # ====== 方式三：多股票拼大图训练（所有股票拼成一个大图，共用一个模型） ======
    #run_method_three(compare_modes_multi=['onlyGAT'], ifSaveModel=True)  # 单模式训练，自动从沪深300取前 maxStockCount 只股票
    #run_method_three(compare_modes_multi=['mixed', 'onlyGAT'], ifSaveModel=True)  # 对比多种网络模式，训练完保存模型
    #run_method_three(stock_list_multi=['000009.SZ', '000010.SZ'], compare_modes_multi=['onlyGAT'], ifSaveModel=True)  # 手动指定股票列表

    # ====== 方式四：加载已保存模型直接预测（不训练，用于测试已训练好的模型） ======
    #run_method_four(model_name='20250101_onlyGAT', net_mode='onlyGAT')  # 加载模型预测，自动从沪深300取前 maxStockCount 只股票
    #run_method_four(model_name='20250101_mixed', net_mode='mixed', stock_list=['600519.SH', '601318.SH', '601398.SH'])  # 手动指定股票，用指定股票的数据做预测
    #run_method_four(model_name='20260801_onlyGAT_50stocks', net_mode='onlyGAT', stock_list=['600519.SH', '601318.SH', '601398.SH'])  # 手动指定股票，用指定股票的数据做预测
    #run_method_four(model_name='20260821_onlyGAT_300stocks', net_mode='onlyGAT', stock_list=['000001.SZ', '000002.SZ', '000003.SZ'])  # 手动指定股票，用指定股票的数据做预测

    # ====== 方式五：滚动预测（模拟真实交易场景，逐天预测） ======
    #run_method_four_rolling(model_name='20250101_onlyGAT', net_mode='onlyGAT', stock_list=['600519.SH', '601318.SH', '601398.SH'])  # 手动指定股票，滚动预测
    #run_method_four_rolling(model_name='20250101_onlyGAT', net_mode='onlyGAT')  # 自动从沪深300取前 maxStockCount 只股票，滚动预测

    # ====== 方式六：实盘/回测预测（可切换模式） ======
    # 使用方法：
    #   1. 先用方式三训练并保存模型（如训练日期为20260824，保存模型名为 20260824_onlyGAT）
    #   2. 设置 dataDate = '2026-08-24'
    #   3. 调用 run_live_predict，通过 is_live 参数切换模式：
    #      - is_live=True（默认）: 实盘预测，用截止到今天的数据预测明天，结果保存到 txt 文件
    #      - is_live=False: 回测验证，用截止到昨天的数据预测今天，评估模型准确率
    #   4. stock_list 可选：
    #      - 手动指定：如 ['600519.SH', '601318.SH']，只预测指定股票
    #      - 留空或不传：自动从沪深300码表获取全部股票

    # 实盘模式 + 手动指定股票：预测明天（2026-08-25）的买卖信号，结果保存到 实盘预测_20260824.txt
    #run_live_predict(model_name='20260824_onlyGAT', net_mode='onlyGAT', stock_list=['600519.SH', '601318.SH', '601398.SH'], is_live=True)

    # 实盘模式 + 自动获取沪深300：预测明天，所有沪深300股票的信号保存到 txt
    #run_live_predict(model_name='20260824_onlyGAT', net_mode='onlyGAT', is_live=True)

    # 实盘模式 + 自动获取前50只：只预测沪深300中前50只股票
    #run_live_predict(model_name='20260824_onlyGAT', net_mode='onlyGAT', is_live=True, max_count=50)

    # 回测模式 + 手动指定股票：预测今天（2026-08-24）的买卖信号，和真实标签对比评估准确率
    #run_live_predict(model_name='20260821_onlyGAT_300stocks', net_mode='onlyGAT', stock_list=['600519.SH', '601318.SH', '601398.SH'], is_live=False)

    # 回测模式 + 自动获取沪深300：评估模型在全部沪深300股票上的准确率
    #run_live_predict(model_name='20260821_onlyGAT_300stocks_99.34', net_mode='onlyGAT', is_live=False)
    




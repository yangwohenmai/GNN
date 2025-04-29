"""
构建出监督学习的标签数据
1.构建一个移动窗口
2.遍历交易序列，取信号转折点
3.构建移动窗口->信号转折点对应关系，即X->Y
"""
import Strategy_BLJJ
import time
import tushare as ts
import sys
import datetime as dt
import typing

from Strategy import CrossSimple
from Strategy import TradeTag

sys.path.append('..\..')
from Helper import CsvHelper
from Helper import LogHelper
from Helper import SqliteHelper
from Helper import DataConversion
from Helper import DrawHelper

"""
根据不同的策略结果生成相应的监督学习标签
"""
# 获取所有股票列表
def GetAllStockList():
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    try:
        pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
        for col in df.values:
            allstocklistdic[col[0]] = col[1]
            allstocklistdic1[col[1]] = col[0]
        return allstocklistdic
    except Exception as e:
        print(str(e))
        allstocklistdic["600000.SH"] = "600000.SH"
        return allstocklistdic
    

# 获取股票行情数据
def GetStockData(stockCode, period=1401, calDay=100, type="D", benchmark="close"):
    startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y%m%d")
    df = ts.pro_bar(ts_code=stockCode, adj='qfq', freq=type, start_date=startdate, end_date=time.strftime("%Y%m%d"))
    #df = ts.pro_bar(ts_code="300083.SZ", start_date=startdate, end_date="20220121")
    
    # 样本点小于40个不计算
    if df is None or df.values is None or len(df.values) < 40:
        print("数据缺失：",stockCode)
        return 0
    
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.values)-1,-1,-1):
        value = df.values[i]
        OrderDic[value[1]] = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5], \
            'lclose':value[6], 'change':value[7], 'chg':value[8], 'vol':value[9], 'amount':value[10]*1000}
    return OrderDic

# 生成二元Normal标签（不区分转折日）
# 不区分指标转折日期，只按时间顺序取窗口标签
# 上涨趋势的区间数据标记1，下跌趋势的区间数据标记0
# period:要取行情数据的长度周期，向前取period个自然日数据
# calDay:要计算的长度天数，取最近calDay个自然日计算
# window:窗口大小
def CalNormal(code, period=700, calDay=300, window=3):
    benchmark = "close"
    OrderDic = GetStockData(code, period, calDay, "D", benchmark)
    # 获取BLJJ策略结果
    resultBLJJ = Strategy_BLJJ.MainFunc(code, period, calDay, "D", benchmark)["BLJJDic"]
    if resultBLJJ == False:
        return False
    # 根据结果获取信号状态区间
    buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'], resultBLJJ['longList'], resultBLJJ['shortList'])
    # 变化标记
    tempVariable = -1
    # 窗口序列
    windowList = list()
    # 标签集合
    tagList = list()
    # 窗口大小，由于数组最后一位作为y，所以要多取一位
    window = window + 1
    for key,value in buyAndSellPeriod["flagDic"].items():
        # 记录窗口序列
        if key in OrderDic:
            OrderDic[key]["long"] = buyAndSellPeriod["longDic"][key]
            OrderDic[key]["short"] = buyAndSellPeriod["shortDic"][key]
            windowList.append(OrderDic[key])
        if value == -1 or len(windowList)< window:
            continue
        tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":value,"long":buyAndSellPeriod["longDic"][key],"short":buyAndSellPeriod["shortDic"][key]})
    
    return tagList


# 生成二元标签
# 只记录转折点当日及其窗口数据,非转折日期数据不记录
# 如果当日是买卖转折点，记转折标签，转涨记1，转跌记0
# period:要取行情数据的长度周期，向前取period个自然日数据
# calDay:要计算的长度天数，取最近calDay个自然日计算
def CalTwo(code, period=700, calDay=300, window=3):
    benchmark = "close"
    OrderDic = GetStockData(code, period, calDay, "D", benchmark)
    # 获取BLJJ策略结果
    resultBLJJ = Strategy_BLJJ.MainFunc(code, period, calDay, "D", benchmark)["BLJJDic"]
    # 根据结果获取信号状态区间
    buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'])
    # 变化标记
    tempVariable = -1
    # 窗口序列
    windowList = list()
    # 标签集合
    tagList = list()
    # 窗口大小，由于数组最后一位作为y，所以要多取一位
    window = window + 1
    for key,value in buyAndSellPeriod.items():
        # 记录窗口序列
        windowList.append(OrderDic[key])
        if value == -1 or len(windowList)< window:
            continue
        if tempVariable != value:
            tempVariable = value
            tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":value})
            #LogHelper.text_create("log", "{0},{1}".format(code,tagList[len(tagList)-1]), '')
            pass
        else:
            pass
    
    return tagList

# 生成三元标签
# 如果当日是买卖转折点，记转折标签，转涨记1，转跌记0
# 如果当日是走势延续，记延续标签，涨跌延续均记2
# period:要取行情数据的长度周期，向前取period个自然日数据
# calDay:要计算的长度天数，取最近calDay个自然日计算
def CalThree(code, period=700, calDay=300, window=3):
    benchmark = "close"
    OrderDic = GetStockData(code, period, calDay, "D", benchmark)
    # 获取BLJJ策略结果
    resultBLJJ = Strategy_BLJJ.MainFunc(code, period, calDay, "D", benchmark)["BLJJDic"]
    # 根据结果获取信号状态区间
    buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'])
    print(buyAndSellPeriod)
    # 变化标记
    tempVariable = -1
    # 窗口序列
    windowList = list()
    # 标签集合
    tagList = list()
    # 窗口大小，由于数组最后一位作为y，所以要多取一位
    window = window + 1
    for key,value in buyAndSellPeriod.items():
      # 记录窗口序列
        windowList.append(OrderDic[key])
        if value == -1 or len(windowList)<= window:
            continue
        # 如果是买卖转折点，记转折标签，转涨记1，转跌记0
        if tempVariable != value:
            tempVariable = value
            tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":value})
        # 如果是走势延续，记延续标签，涨跌延续均记2
        else:
            tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":2})
    return tagList
    
# 生成四元标签
# 如果当日是买卖转折点，记转折标签，转涨记1，转跌记0
# 如果当日是走势延续，记延续标签,涨延续记2,跌延续记3
# period:要取行情数据的长度周期，向前取period个自然日数据
# calDay:要计算的长度天数，取最近calDay个自然日计算
# window:窗口大小
def CalFour(code, period=700, calDay=300, window=3):
    benchmark = "close"
    OrderDic = GetStockData(code, period, calDay, "D", benchmark)
    # 获取BLJJ策略结果
    resultBLJJ = Strategy_BLJJ.MainFunc(code, period, calDay, "D", benchmark)["BLJJDic"]
    # 根据结果获取信号状态区间
    buyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultBLJJ['tList'], resultBLJJ['buyDateDic'], resultBLJJ['sellDateDic'], resultBLJJ['longList'], resultBLJJ['shortList'])
    # 变化标记
    tempVariable = -1
    # 窗口序列
    windowList = list()
    # 标签集合
    tagList = list()
    # 窗口大小，由于数组最后一位作为y，所以要多取一位
    window = window + 1
    # 对每个code生成对应的数据标签
    for key,value in buyAndSellPeriod["flagDic"].items():
        # 记录窗口序列
        if key in OrderDic:
            OrderDic[key]["long"] = buyAndSellPeriod["longDic"][key]
            OrderDic[key]["short"] = buyAndSellPeriod["shortDic"][key]
            windowList.append(OrderDic[key])
        if value == -1 or len(windowList)< window:
            continue
        # 如果是买卖转折点，记转折标签，转涨记1，转跌记0
        if tempVariable != value:
            tempVariable = value
            tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":value,"long":buyAndSellPeriod["longDic"][key],"short":buyAndSellPeriod["shortDic"][key]})
        # 如果是走势延续，记延续标签,涨延续记2,跌延续记3
        else:
            if value == 0:
                tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":2,"long":buyAndSellPeriod["longDic"][key],"short":buyAndSellPeriod["shortDic"][key]})
            else:
                tagList.append({"tdate":key,"windowList":windowList[-window:-1],"signal":3,"long":buyAndSellPeriod["longDic"][key],"short":buyAndSellPeriod["shortDic"][key]})
    return tagList

# 主函数
if __name__ == '__main__':
    resultList = list()
    for code in GetAllStockList().keys():
        try:
            tagList = CalNormal(code)
            #tagList = CalFour(code)
        except Exception as e:
            LogHelper.text_create("error", "{0},{1}".format(code,e), '')
            time.sleep(65)
            #resultList.append(Strategy_BLJJ.MainFunc(code, 700, 300, "D", benchmark)["BLJJDic"])
            tagList = CalFour(code)
        else:
            pass
        LogHelper.text_create(code, "{0},{1}".format(code,tagList), '')
        #LogHelper.text_create("log", "{0},{1}".format(code,tagList[len(tagList)-1]), '')
    print("Finish")
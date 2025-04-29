from asyncio.windows_events import NULL
import numpy as np
import tushare as ts
import sys
import os
import datetime as dt
import time
import typing

from Strategy import CrossSimple
from Strategy import BLJJ
from Strategy import TradeTag

sys.path.append('..')
from Helper import CsvHelper
from Helper import LogHelper
from Helper import SqliteHelper
from Helper import DataConversion
from Helper import DrawHelper

# 定位文件路径
filePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(filePath)




# 策略主函数
# period:要取值的长度周期，向前取period个自然日数据
# calDay:要计算的长度天数，取最近calDay个自然日计算
# type:周期级别，D/W/M 日/周/月
# benchmark:买卖点交易基准，close以收盘价成交，benchmarkList以当天特定价成交
def MainFunc(stockCode, period=1401, calDay=100, type="D", benchmark="close"):
    #datax = CsvHelper.GetAllDataFloatWithHeader(filePath + r'\data\BLJJData.csv')
    startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y%m%d")
    df = ts.pro_bar(ts_code=stockCode, adj='qfq', freq=type, start_date=startdate, end_date=time.strftime("%Y%m%d"))
    #df = ts.pro_bar(ts_code="300083.SZ", start_date=startdate, end_date="20220121")
    
    # 样本点小于40个不计算
    if df is None or df.values is None or len(df.values) < calDay:
        print("数据缺失：",stockCode, "，最大数据量：", len(df.values))
        return {"BLJJDic":False}
    
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.values)-1,-1,-1):
        value = df.values[i]
        OrderDic[value[1]] = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5]}
    
    # BLJJ，计算长短周期交叉线的数值，
    resultBLJJ = BLJJ.BLJJ(OrderDic,calDay)

    if benchmark == "close":
        # 买卖点交易基准（取收盘价）
        benchmarkList = resultBLJJ["closeList"]
    elif benchmark == "benchmark":
        # 买卖点交易基准（取中间价）
        benchmarkList = resultBLJJ["benchmarkList"]
    
    #benchmarkList = datax.loc[:,'C']
    #for i in range(len(datax.loc[:,'C'])):
        #benchmarkList.append((datax.loc[:,'C'][i]+datax.loc[:,'O'][i])/2)
    
    # 简单交叉算法计算买卖点
    resultCrossDic = CrossSimple.CrossSimple(resultBLJJ["tList"], resultBLJJ["longList"], resultBLJJ["shortList"], benchmarkList, resultBLJJ["closeList"])
    #redic = CrossSimple.CrossSimple(datax.loc[:,'T'], datax.loc[:,'Long'], datax.loc[:,'Short'], benchmarkList, datax.loc[:,'C'])
    #print("交易信息:", resultCrossDic["tradeInfo"])
    #print("交易日期表:",resultCrossDic["tList"])
    #print("买入日期:", resultCrossDic["buyDateDic"])
    #print("卖出日期:", resultCrossDic["sellDateDic"])
    #print("收益曲线:", resultCrossDic["myReturnList"])
    #print("自然涨跌曲线:", resultCrossDic["ObjectiveRetrunList"])


    # 根据买卖点标记持仓周期标签
    #BuyAndSellPeriod = TradeTag.TimeLineBuyAndSellPeriod(resultCrossDic['tList'], resultCrossDic['buyDateDic'], resultCrossDic['sellDateDic'])

    # 收益统计，若从来没出现过交易信号，输出0
    signalDic = dict()
    if len(resultCrossDic["myReturnList"]) > 0:
        str = resultCrossDic["tradeInfo"][resultCrossDic["tradeInfo"].find("收益率："):]
        signalDic["info"] = str.replace("\r\n",",").replace("%","").replace("收益率：","").replace("自然涨跌：","").replace(">",",")
        signalDic["BLJJDic"] = resultCrossDic
        return signalDic
    else:
        signalDic["info"] = 0
        signalDic["BLJJDic"] = NULL
        return signalDic

    # 画图，取收盘价和收益率曲线（包含初始价格），作图进行比较
    CrossLineDic = dict()
    CrossLineDic['C'] = resultBLJJ["closeList"]
    CrossLineDic['C'] = resultCrossDic["ObjectiveRetrunList"]# ObjectiveRetrunList默认就是closeList
    CrossLineDic['Win'] = resultCrossDic["myReturnList"]

    #draw = DrawHelper.DrawLineFunc()
    #draw.DrawLine_WithSignal(resultCrossDic["tList"], CrossLineDic, resultCrossDic["buyDateDic"], resultCrossDic["sellDateDic"],stockCode)
    DrawHelper.DrawLine_WithSignal(resultCrossDic["tList"], CrossLineDic, resultCrossDic["buyDateDic"], resultCrossDic["sellDateDic"],stockCode)
    input()
    



# 获取所有股票列表
def GetAllStockList():
    #pro = ts.set_token('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[1]
        allstocklistdic1[col[1]] = col[0]
    return allstocklistdic


# 调用方式
if __name__ == '__main__':
    resultList = list()
    for code in GetAllStockList().keys():
        try:
            resultList.append(MainFunc(code, 700, 300, "D", "benchmark")["info"])
        except Exception as e:
            LogHelper.text_create("error", "{0},{1}".format(code,e), '')
            time.sleep(65)
            resultList.append(MainFunc(code, 700, 300, "D", "benchmark")["info"])
        else:
            pass
        LogHelper.text_create("log", "{0},{1}".format(code,resultList[len(resultList)-1]), '')
    print("Finish")
    






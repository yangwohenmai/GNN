import tushare as ts
import datetime as dt
import time
import typing
import sys
import os
import baostock as bs
import math

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import pandas as pd

sys.path.append('..\..')
#print(sys.path)
# 打印文件绝对路径（absolute path）
#print (os.path.abspath(__file__))  
# 打印文件的目录路径（文件的上两层目录）, 这个时候是在 atm 这一层。就是os.path.dirname这个再用了一次
#print (os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))) 
# 要调取其他目录下的文件。 需要在atm这一层才可以
BASE_DIR=  os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#print(BASE_DIR)
# 将这个路径添加到环境变量中。
sys.path.append( BASE_DIR  )




""""
# 寻找反转模式
# 目标：出死叉后，判断死叉点后chgDayCount日最大涨幅是否达到预期，达到预期则选出来
# 1. ma_short 连续在 ma_long 下 n 天后，首次出现交叉，获取这样的区间数据（不包含交叉当天）
# 2. 找到在交叉点对应的日期
# 3. 从交叉点向未来取p个交易日，判断从最低位起，涨幅是否达到x%，若达到则为所求
# 4. 从交叉点向历史取 m 个交易日的数据，用于作图
# 5. 记录股票代码、日期区间、价格区间，输出区间数据的图像

# 后续改进：
目标：调整对出现交叉后，后续趋势评判的标准
# 1.若连续 m 天 ma_short > ma_long，即为所求
# 2.调整交叉前的趋势判断条件；因为连续n天ma_short < ma_long，n若过大，会导致错过很多涨幅满足但下跌趋势不够的股票
"""
codeText = ['000001.SZ','000002.SZ','000063.SZ','000069.SZ','000157.SZ','000425.SZ','000568.SZ','000625.SZ','000651.SZ','000858.SZ','600000.SH','600009.SH','600010.SH','600015.SH','600016.SH','600019.SH','600028.SH','600029.SH']
# 获取所有股票列表
def GetAllStockList1():
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[1]
        allstocklistdic1[col[1]] = col[0]
    return allstocklistdic

def GetAllStockList2():
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    df = pro.bak_daily(trade_date='20240228', fields='trade_date,ts_code,name,close,open')
    #df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[1]
        allstocklistdic1[col[1]] = col[0]
    return allstocklistdic

# 获取所有股票列表
def GetAllStockList():
    # 获取中证500成分股
    rs = bs.query_zz500_stocks()
    # 打印结果集
    zz500_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        zz500_stocks.append(rs.get_row_data())
    df = pd.DataFrame(zz500_stocks, columns=rs.fields)
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        codeArray = col[1].upper().split('.')
        code = codeArray[1]+'.'+codeArray[0]
        allstocklistdic[code] = codeArray[1]
        allstocklistdic1[codeArray[1]] = code
    return allstocklistdic

# 获取股票池列表，isReadText=False则读取文本代码
def GetStockPool(indexCode='399300.SZ', isReadText = False, codeListPath='C:\\MyGit\\ResearchProject\\yangzx\\Temp\\沪深300列表.txt'):
    stockPoolDic = dict()
    try :
        pro = ts.pro_api('1c5440f527d1e513c75d10518ef9fd05a34a33ec4146b353bc7ce5bf')
        # 月初月末各公布一次成分股
        startdate = (dt.datetime.today() - dt.timedelta(31)).strftime("%Y%m%d")
        df = pro.index_weight(index_code=indexCode, start_date=startdate, end_date=time.strftime("%Y%m%d"))
        for col in df.values:
            stockPoolDic[col[1]] = True
    except Exception as e :
        print("异常信息：" + str(e))
    finally :
        if(isReadText is not True) :
            print('获取指定板块失败，获取全市场股票')
            stockPoolDic = GetAllStockList()
        else:
            print('获取指定板块失败，读取指定文本数据')
            fileStr = FileHelper.ReadText(codeListPath)
            codeText = fileStr.replace("'","").split(",")
            stockPoolDic = dict(zip(codeText,codeText))
    return list(stockPoolDic.keys())

# 取股票均线数据
# maPara: 想要获取的均线窗口值
def GetStockMA(stockCode, period=1401, maPara=[10, 20], calDay=100, type="D", benchmark="close"):
    startdate = (dt.datetime.today() - dt.timedelta(period*1)).strftime("%Y%m%d")
    enddate = (dt.datetime.today() - dt.timedelta(period*0)).strftime("%Y%m%d")
    df = ts.pro_bar(ts_code=stockCode, adj='qfq', start_date=startdate, end_date=enddate, ma=maPara)
    
    # 样本点小于40个不计算
    if df is None or df.values is None or len(df.values) < 40:
        print("数据缺失：",stockCode)
        return False
    
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.values)-1,-1,-1):
        # 剔除前n天均线为Nan值的数据
        if i > len(df.values) - maPara[len(maPara)-1]:
            continue
        value = df.values[i]
        OrderDic[value[1]] = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5], \
            'lclose':value[6], 'change':value[7], 'chg':value[8], 'vol':value[9], 'amount':value[10]*1000, \
            'ma_short':value[11], 'ma_v__short':value[12], 'ma_long':value[13], 'ma_v_long':value[14]}
    #print(OrderDic[next(reversed(OrderDic))])
    return OrderDic
# 取股票分钟数据
def GetStockMin(stockCode, startdate, endDate=time.strftime("%Y-%m-%d"), period=1401, calDay=100, type="15", benchmark="close"):
    stockCode = stockCode.split('.')[1].lower()+'.'+stockCode.split('.')[0]
    if startdate == 0:
        startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y-%m-%d")
    else:
        startdate = dt.datetime.strptime(startdate,'%Y%m%d').strftime("%Y-%m-%d")
        endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    #"date,time,code,open,high,low,close,volume,amount,adjustflag"
    df = bs.query_history_k_data_plus(stockCode,"date,time,open,high,low,close,volume",start_date=startdate,end_date=endDate,frequency=type, adjustflag="2")
    # 样本点小于40个不计算
    if df is None or df.error_msg != 'success' or len(df.data) < 40:
        print("数据缺失：",stockCode)
        return False
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.data)):
        value = df.data[i]
        fields = df.fields
        dic = dict()
        for item in range(len(fields)):
            dic[fields[item]] = value[item]
        OrderDic[value[1]] = dic
    return OrderDic

def GetMACD(stockInfo,fast=12,slow=26,signal=9):
    # 记录当前股票的历史信息，用于截取TPO数据
    histroyList = list()
    MACDList = list()
    for stockInfoKey, stockInfoValue in stockInfo.items():
        histroyList.append([stockInfoValue["tdate"],stockInfoValue["open"],stockInfoValue["high"],stockInfoValue["low"],stockInfoValue["close"]])
    stockMACDValue = MACD.MACD_talib(np.array(histroyList)[:,4],fast,slow,signal)
    for i in range(len(histroyList)):
        MACDList.append({'tdate': histroyList[i][0],'price': histroyList[i][1:], 'macd_short':stockMACDValue['MACDDIF'].tolist()[i], 'macd_long':stockMACDValue['MACDDEA'].tolist()[i]})
    return MACDList

# 判断反转模式，本策略以当前金叉/死叉周期中股价的最低点记为趋势反转点。以指标为目标时，可将金叉/死叉点作为趋势反转点
# reversalDay：最低点前，连续金叉/死叉天数
# chgDayCount：从最低点向未来取chgDayCount天作为涨跌幅计算区间
# pchg：反转涨跌幅评判标准
# isCross：是否记录趋势交叉当天的数据信息
def FindReversal(stockMACDList, reversalDay=8, chgDayCount=5, pchg=0.15, TPOlength=25, isCross=True):
    # 连续趋势计数，从最近一次趋势交叉开始计数
    compareCount = 0
    # 全局循环计数器,记录当前趋势交叉点在整个序列中的位置
    globleCount = 0
    # 日期列表
    seriesTdateList = list()
    # 交易数据列表
    seriesList = list()
    orderDic = typing.OrderedDict()
    # 记录当前股票的历史信息，用于截取TPO数据
    histroyList = list()
    for stockMACDValue in stockMACDList:
        histroyList.append(stockMACDValue['price'])
        if math.isnan(float(stockMACDValue['macd_short'])):
            globleCount += 1
            continue
        # 从首次上涨趋势开始时（出现金叉），记录本次趋势内序列
        if float(stockMACDValue['macd_short']) >= float(stockMACDValue['macd_long']):
            seriesTdateList.append(stockMACDValue['tdate'])
            seriesList.append([stockMACDValue['tdate'],stockMACDValue['macd_short'],stockMACDValue['macd_long']])
            compareCount += 1
        # 遇到死叉本轮趋势终止，若本轮上涨趋势满足趋势时长，在orderDic添加本轮趋势序列（默认记录交叉当天信息）
        if float(stockMACDValue['macd_short']) < float(stockMACDValue['macd_long']):
            # 判断死叉前的信息是否满足计算死叉条件
            if compareCount > reversalDay:
                # 判断是否添加死叉当天的信息
                if isCross:
                    seriesTdateList.append(stockMACDValue['tdate'])
                    seriesList.append([stockMACDValue['tdate'],stockMACDValue['macd_short'],stockMACDValue['macd_long']])
                # 找死叉点对应的日期
                if (globleCount + chgDayCount) <= len(stockMACDList) - 1:
                    # 从死叉点向未来取chgDayCount个交易日收盘价，计入chgList，计算涨跌幅
                    chgList = list()
                    for count in range(chgDayCount):
                        # 找到交叉点的位置，从交叉点向未来取chgDayCount天的价格序列
                        chgList.append(stockMACDList[globleCount + count]['price'][3])
                    # 如果跌幅达标 只存储交叉点及历史部分的序列
                    stockPchg = (chgList[0] - min(chgList)) / chgList[0]
                    if stockPchg < pchg:
                        orderDic[stockMACDValue['tdate']] = {'tdateList':seriesTdateList, 'seriesList':seriesList}
                        # 如果历史数据长度满足TPO需求，从交叉点位置向历史取TPOlength天数据
                        if len(histroyList) > TPOlength:
                            orderDic[stockMACDValue['tdate']]['pchg'] = stockPchg
                            # 添加日线数据
                            orderDic[stockMACDValue['tdate']]['TPOList'] = histroyList[-TPOlength:]
                            # 添加分钟数据
                            orderDic[stockMACDValue['tdate']]['startDate'] = stockMACDList[globleCount-24]['tdate']
                            orderDic[stockMACDValue['tdate']]['endDate'] = stockMACDList[globleCount]['tdate']
                            stockMinData = GetStockMin(code,orderDic[stockMACDValue['tdate']]['startDate'],orderDic[stockMACDValue['tdate']]['endDate'])
                            dic = dict()
                            for i in stockMinData.values():
                                dayDate = dt.datetime.strptime(i['date'],'%Y-%m-%d').strftime("%Y%m%d")
                                if dayDate not in dic:
                                    dic[dayDate] = list()
                                dic[dayDate].append([i['high'],i['low'],i['volume']])
                            orderDic[stockMACDValue['tdate']]['TPOMinData'] = dic
            seriesTdateList = list()
            seriesList = list()
            # 完成当前趋势交叉信息提取后，重置趋势计数器，寻找下一次满足条件的交叉
            compareCount = 0
        globleCount += 1
    return orderDic

# 画出每个符合要求的图
def SavePicture(code, dataDic, reversalDay, reversalDayOffset, dirPath=''):
    if dirPath == '' or not os.path.exists(dirPath):
        dirPath = sys.path[0]
        print("缺少保存路径，默认保存至当前文件夹下：" + dirPath)
    for item in dataDic.values():
        if len(item["tdateList"]) > reversalDay - reversalDayOffset and 'TPOList' in item.keys() :
            path = "{}\\{}_{}.png".format(dirPath, code, item["endDate"])
            DrawHelperTPOMinVol.DrawTPOImage(item['TPOMinData'],[],5,5,path,{'high':0,'low':1,'vol':2})
            #DrawTPOMinVolAllHelper.DrawTPOImage(item['TPOMinData'],[],25,1,path,{'high':0,'low':1,'vol':2})
            #DrawTPOMinVolSingleDayHelper.DrawTPOImage(item['TPOMinData'],[],25,1,path,{'high':0,'low':1,'vol':2})
        else:
            print(code,":数据量不足","，不生成图片")

def test():
    return 'FuncDown'
# 主函数
if __name__ == '__main__':
    GetAllStockList2()
    now = dt.datetime.now()
    print('123'+str(dt.datetime.now()))

    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond error_msg:'+lg.error_msg)
    #allstocklistdic = GetAllStockList()
    a = 0
    b = 0
    sampleCount = 50
    

    for code in GetAllStockList().keys():
        a = a + 1
        if a > 3:
            continue
        try:
            # 获取均线数据
            stockMADic = GetStockMA(code)
            if stockMADic == False:
                print(code+"行情获取失败")
                continue
            elif len(stockMADic) < sampleCount:
                print(code+"低于最小样本限制")
                continue
            else:
                stockMACDDic = GetMACD(stockMADic,12,26,9)
                print(len(stockMACDDic))
                print(stockMACDDic[0])
                for key in stockMACDDic[0]:
                    print(key + ':' + str(stockMACDDic[0][key]))
        except Exception as ex:
            print("失败代码："+code+"，异常信息："+str(ex))
            for item in GetAllStockList().keys():
                a = a + 1
                if a < 5:
                    print(item)
    print('Finish')
    
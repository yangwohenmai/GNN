import tushare as ts
import datetime as dt
import time
import typing
import sys
import os
import baostock as bs
import math
import 获取股市码表

import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
from collections import deque
import pandas as pd

sys.path.append('..\..')
print(sys.path)
# 打印文件绝对路径（absolute path）
print (os.path.abspath(__file__))  
# 打印文件的目录路径（文件的上两层目录）, 这个时候是在 atm 这一层。就是os.path.dirname这个再用了一次
print (os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))) 
# 要调取其他目录下的文件。 需要在atm这一层才可以
BASE_DIR=  os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
print(BASE_DIR)
# 将这个路径添加到环境变量中。
sys.path.append( BASE_DIR  )
#from Helper import FileHelper


# 取股票均线数据
# maPara: 想要获取的均线窗口值
# period: 想要获取的数据周期
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

# 取股票行情数据
# maPara: 想要获取的均线窗口值
# period: 想要获取的数据周期
def GetStockPriceTushare(stockCode, period=1401, maPara=[10, 20], calDay=100, type="D", benchmark="close"):
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
            'lclose':value[6], 'change':value[7], 'chg':value[8], 'vol':value[9], 'amount':value[10]*1000}
    #print(OrderDic[next(reversed(OrderDic))])
    return OrderDic

# 获取股票(日周月)行情数据
# stockCode: 示例sh.600000
# fields: 返回列字段，分钟数据与日月周数据略有不同，详见文档
# period: 数据周期
# type: 数据类型，默认为d，日k线；d=日k线、w=周、m=月，不区分大小写
# adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
def GetStockPriceDWMBaostock(stockCode, startdate, endDate=time.strftime("%Y-%m-%d"), period=1401, calDay=100, type="d", benchmark="close"):
    stockCode = stockCode.split('.')[1].lower()+'.'+stockCode.split('.')[0]
    if startdate == 0:
        startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y-%m-%d")
    else:
        startdate = dt.datetime.strptime(startdate,'%Y%m%d').strftime("%Y-%m-%d")
        endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    #"code,date,open,high,low,close,volume,amount,adjustflag"
    df = bs.query_history_k_data_plus(stockCode,"code,date,open,high,low,close,volume,pctChg",start_date=startdate,end_date=endDate,frequency=type, adjustflag="2")
    # 样本点小于40个不计算
    if df is None or df.error_msg != 'success' or len(df.data) == 0:
        print("获取行情数据数据异常：",stockCode)
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

# 获取股票(分钟)行情数据
# stockCode: 示例sh.600000
# fields: 返回列字段，分钟数据与日月周数据略有不同，详见文档
# period: 数据周期
# type: 5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
# adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
def GetStockPriceMinBaostock(stockCode, startdate, endDate=time.strftime("%Y-%m-%d"), period=1401, calDay=100, type="d", benchmark="close"):
    stockCode = stockCode.split('.')[1].lower()+'.'+stockCode.split('.')[0]
    if startdate == 0:
        startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y-%m-%d")
    else:
        startdate = dt.datetime.strptime(startdate,'%Y%m%d').strftime("%Y-%m-%d")
        endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    #"date,time,code,open,high,low,close,volume,amount,adjustflag"
    df = bs.query_history_k_data_plus(stockCode,"date,time,code,open,high,low,close,volume,amount",start_date=startdate,end_date=endDate,frequency=type, adjustflag="2")
    # 样本点小于40个不计算
    if df is None or df.error_msg != 'success' or len(df.data) == 0:
        print("获取行情数据数据异常：",stockCode)
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

# 主函数
if __name__ == '__main__':
    print('begin'+str(dt.datetime.now()))
    sampleCount = 50
    dataCount = 0
    #### 登陆系统 ####
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond error_msg:'+lg.error_msg)
    #GetALLStockListBaostock()
    #GetAllStockListTushare()
    #GetAllStockListTushareBak()

    stockPoolList = 获取股市码表.GetStockPool('',False,'')
    
    for code in 获取股市码表.GetALLStockListBaostock().keys():
        if len(stockPoolList) == 0 or code not in stockPoolList:
            continue
        try:
            # 获取行情数据
            stockPriceDic = GetStockPriceDWMBaostock(code, 0)
            if stockPriceDic == False:
                print(code+"行情获取失败")
                continue
            elif len(stockPriceDic) < sampleCount:
                print(code+"低于最小样本限制")
                continue
            else:
                dataCount += 1
                print(code+'已输出,序号:NO.'+str(dataCount))
            if dataCount == 500:
                time.sleep(60)
        except Exception as ex:
            print("失败代码："+code+"，异常信息："+str(ex))
    print("finish")
    input()
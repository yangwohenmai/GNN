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



# 获取当日所有股票列表
def GetAllStockListTushare():
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[1]
        allstocklistdic1[col[1]] = col[0]
    return allstocklistdic

# 获取当日所有股票列表 备用行情 https://tushare.pro/document/2?doc_id=255
def GetAllStockListTushareBak():
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    trade_date = (dt.datetime.today() - dt.timedelta(1)).strftime("%Y%m%d")
    df = pro.bak_daily(trade_date=trade_date, fields='ts_code,trade_date,name')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[0].split('.')[0]
        allstocklistdic1[col[0].split('.')[0]] = col[0]
    return allstocklistdic

# 获取上证50成分股票列表（baostock）
def GetSZ50StockListBaostock():
    # 获取沪深300成分股
    rs = bs.query_sz50_stocks()
    # 打印结果集
    sz50_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        sz50_stocks.append(rs.get_row_data())
    df = pd.DataFrame(sz50_stocks, columns=rs.fields)
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        codeArray = col[1].upper().split('.')
        code = codeArray[1]+'.'+codeArray[0]
        allstocklistdic[code] = codeArray[1]
        allstocklistdic1[codeArray[1]] = code
    return allstocklistdic

# 获取沪深300成分股票列表（baostock）
def GetHS300StockListBaostock():
    # 获取沪深300成分股
    rs = bs.query_hs300_stocks()
    # 打印结果集
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        hs300_stocks.append(rs.get_row_data())
    df = pd.DataFrame(hs300_stocks, columns=rs.fields)
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        codeArray = col[1].upper().split('.')
        code = codeArray[1]+'.'+codeArray[0]
        allstocklistdic[code] = codeArray[1]
        allstocklistdic1[codeArray[1]] = code
    return allstocklistdic

# 获取中证500成分股票列表（baostock）
def GetZZ500StockListBaostock():
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

# 获取当日所有股票列表（baostock）
# 缺少包含：not like '900%' and  SYMBOL not like '200%' and  SYMBOL not like '299%'  and  SYMBOL not in(201872)
# 多余包含：like '399%' and like 'sh.000%'
def GetALLStockListBaostock():
    # 获取中证500成分股
    rs = bs.query_all_stock(day='2024-03-01')
    allDataList = []
    while (rs.error_code == '0') & rs.next():
        code = rs.get_row_data()
        if code[0][:6] != 'sh.000' and code[0][:6] != 'sz.399':
            allDataList.append(code)
    df = pd.DataFrame(allDataList, columns=rs.fields)
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        codeArray = col[0].upper().split('.')
        code = codeArray[1]+'.'+codeArray[0]
        allstocklistdic[code] = codeArray[1]
        allstocklistdic1[codeArray[1]] = code
    return allstocklistdic

# 获取股票池列表，isReadText=False则读取文本代码
def GetStockPool(indexCode='399300.SZ', isReadText = False, codeListPath='C:\\MyGit\\ResearchProject\\yangzx\\Temp\\沪深300列表.txt'):
    stockPoolDic = dict()
    try :
        pro = ts.pro_api('1c5440f527d1e513c75d10518ef9fd05a34a33ec4146b353bc7ce5bf')
        # 月初月末各公布一次成分股pro.index_weight
        startdate = (dt.datetime.today() - dt.timedelta(31)).strftime("%Y%m%d")
        df = pro.index_weight(index_code=indexCode, start_date=startdate, end_date=time.strftime("%Y%m%d"))
        for col in df.values:
            stockPoolDic[col[1]] = True
    except Exception as e :
        print("异常信息：" + str(e))
    finally :
        if(isReadText is not True) :
            print('调用tushare指定板块接口失败，调用baostock股票接口')
            stockPoolDic = GetHS300StockListBaostock()
            #stockPoolDic = GetAllStockListTushare()
            #stockPoolDic = GetAllStockListTushareBak()
        else:
            print('调用指定板块接口失败，读取指定文本数据')
            fileStr = FileHelper.ReadText(codeListPath)
            codeText = fileStr.replace("'","").split(",")
            stockPoolDic = dict(zip(codeText,codeText))
    return list(stockPoolDic.keys())


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
    # 获取股票池 上证50:000016.SH 沪深300:399300.SZ 上证180：000010.SH
    stockPoolList = GetStockPool('',False,'')
    
    for code in GetALLStockListBaostock().keys():
        if len(stockPoolList) == 0 or code not in stockPoolList:
            continue
        try:
            if len(stockPoolList) < sampleCount:
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
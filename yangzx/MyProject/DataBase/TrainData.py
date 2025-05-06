import tushare as ts
import datetime as dt
import time
import typing
import sys
import os
import baostock as bs
import math


import numpy as np
from collections import deque
import pandas as pd

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# 对个股构建全局图
def TrainData(stockPriceDic):
    # 有向图，每条数据只和过去7天有关系
    # 每只股票所有历史数据构成一个全局图
    # 构建关系矩阵 1.特征矩阵 2.节点关系矩阵 3.权重矩阵
    n = 7
    list1 = list()
    list2 = list()
    #for i in range(0,n):
    #    list1.append(i)
    #    list2.append(i+1)
    #list3 = list()
    #list3.append(list1)
    #list3.append(list2)
    #edge_index = torch.tensor(np.array(list3))
    #print(edge_index)

    count = len(stockPriceDic)
    for i in range(0,count):
        for j in range(n,0,-1):
            if i < n:
                continue
            else:
                list1.append(i-j)
                list2.append(i)
    list3 = list()
    list3.append(list1)
    list3.append(list2)
    edge_index = torch.tensor(np.array(list3))
    print(edge_index)
    
    dataListx = list()
    dataListy = list()
    data = list()
    #for key,f in stockPriceDic.items():
    #    dataListx.append([float(f['open']),float(f['close']),float(f['low']),float(f['high']),float(f['volume'])])
    #    dataListy.append(0 if float(f['pctChg']) < 0 else 1)
    #    if len(dataListx) < n:
    #        continue
    #    else:
    #        data.append(Data(x=torch.tensor(np.array(dataListx[-n:])),y=torch.tensor(np.array(dataListy[-1:])),edge_index=edge_index))
    
    for key,f in stockPriceDic.items():
        dataListx.append([float(f['open']),float(f['close']),float(f['low']),float(f['high']),float(f['volume'])])
        dataListy.append(0 if float(f['pctChg']) < 0 else 1)

    data.append(Data(x=torch.tensor(np.array(dataListx)),y=torch.tensor(np.array(dataListy)),edge_index=edge_index))
    return data

def TrainDataInt(stockPriceDic):
    # 根据N来截断数据N=7
    # 构建关系矩阵 1.特征矩阵 2.节点关系矩阵 3.权重矩阵
    n = 1
    list1 = list()
    list2 = list()
    count = len(stockPriceDic)
    for i in range(0,count):
        for j in range(n,0,-1):
            if i < n:
                continue
            else:
                list1.append(i-j)
                list2.append(i)
    list3 = list()
    list3.append(list1)
    list3.append(list2)
    edge_index = torch.tensor(np.array(list3))
    #print(edge_index)
    
    dataListx = list()
    dataListy = list()
    data = list()
    dayCount = 0
    for key,f in stockPriceDic.items():
        dayCount += 1
        dataListx.append([float(f['open']),float(f['close']),float(f['low']),float(f['high']),float(f['pctChg']),dayCount/len(stockPriceDic),0 if float(f['pctChg']) < 0 else 1])
        dataListy.append(0 if float(f['pctChg']) < 0 else 1)

    data.append(Data(x=torch.tensor(np.array(dataListx)),y=torch.tensor(np.array(dataListy)),edge_index=edge_index))
    return data

def TrainDataMACD(stockPriceDic):
    # 根据N来截断数据N=7
    # 构建关系矩阵 1.特征矩阵 2.节点关系矩阵 3.权重矩阵
    n = 1
    list1 = list()
    list2 = list()
    count = len(stockPriceDic)
    for i in range(0,count):
        for j in range(n,0,-1):
            if i < n:
                continue
            else:
                list1.append(i-j)
                list2.append(i)
    list3 = list()
    list3.append(list1)
    list3.append(list2)
    edge_index = torch.tensor(np.array(list3))
    #print(edge_index)
    
    dataListx = list()
    dataListy = list()
    data = list()
    dayCount = 0
    for key,f in stockPriceDic.items():
        dayCount += 1
        dataListx.append([float(f['open']),float(f['close']),float(f['low']),float(f['high']),float(f['pctChg']),dayCount/len(stockPriceDic),f['flag']])
        dataListy.append(f['flag'])

    data.append(Data(x=torch.tensor(np.array(dataListx)),y=torch.tensor(np.array(dataListy)),edge_index=edge_index))
    return data

# 主函数
if __name__ == '__main__':
    import StockPool
    import StockData
    arr = [[1,2,3,4],[2,3,4,5]]
    arr = np.array(arr)
    print("ndarray的数据类型：", arr.dtype)
    t= torch.tensor(arr)
    print(t)
    data = Data(x=t,y=t,edge_index=t)
    a = LabelEncoder()
    data1 = [3, 2, 3, 2, 5]
    b = np.array(data1)
    c = a.fit_transform(b)
    print(c)




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

    stockPoolList = StockPool.GetStockPool('',False,'')
    
    for code in StockPool.GetALLStockListBaostock().keys():
        if len(stockPoolList) == 0 or code not in stockPoolList:
            continue
        try:
            # 获取行情数据
            stockPriceDic = StockData.GetStockPriceDWMBaostock(code, 0)
            if stockPriceDic == False:
                print(code+"行情获取失败")
                continue
            elif len(stockPriceDic) < sampleCount:
                print(code+"低于最小样本限制")
                continue
            else:
                TrainData(stockPriceDic)
                dataCount += 1
                print(code+'已输出,序号:NO.'+str(dataCount))
            if dataCount == 500:
                time.sleep(60)
        except Exception as ex:
            print("失败代码："+code+"，异常信息："+str(ex))
    print("finish")
    input()
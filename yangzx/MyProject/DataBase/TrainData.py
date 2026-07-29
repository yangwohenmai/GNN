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
    edge_index = torch.tensor(np.array(list3), dtype=torch.long)
    #print(edge_index)
    
    dataListx = list()
    dataListy = list()
    data = list()
    dayCount = 0
    for key,f in stockPriceDic.items():
        dayCount += 1
        # 注意：flag 保留供邻居节点通过图边获取历史信号，但残差路径不引用任何当天字段（见Net.forward）
        dataListx.append([float(f['open']),float(f['close']),float(f['low']),float(f['high']),float(f['pctChg']),dayCount/len(stockPriceDic),f['flag']])
        dataListy.append(f['flag'])

    data.append(Data(x=torch.tensor(np.array(dataListx)),y=torch.tensor(np.array(dataListy)),edge_index=edge_index))
    return data

def TrainDataMACDWindowK(stockPriceDic, edgeWindowK=3, edgeStride=1):
    # K窗口入边版本（新函数，老函数TrainDataMACD保留用于对比实验）
    # 每个节点i接收前edgeWindowK个相邻节点的边：X[i-K]~X[i-1] -> X[i]
    # edgeWindowK=1时等价于TrainDataMACD的单链结构
    # edgeStride控制边的稀疏性：从最近的X[i-1]开始每隔stride取一个，即偏移j∈{1, 1+s, 1+2s, ...}且j≤K
    # 例：K=3、stride=2时 j∈{1,3}，即X[i-3]、X[i-1] -> X[i]；stride=1为稠密窗口（原行为）
    # 开头节点采用部分窗口（有多少历史连多少），避免无入边节点在add_self_loops=False下卷积输出全零
    # 构建关系矩阵 1.特征矩阵 2.节点关系矩阵 3.权重矩阵
    list1 = list()
    list2 = list()
    count = len(stockPriceDic)
    for i in range(1, count):
        for j in range(min(i, edgeWindowK), 0, -1):
            if (j - 1) % edgeStride != 0:
                continue
            list1.append(i-j)
            list2.append(i)
    list3 = list()
    list3.append(list1)
    list3.append(list2)
    edge_index = torch.tensor(np.array(list3), dtype=torch.long)
    #print(edge_index)

    dataListx = list()
    dataListy = list()
    data = list()
    dayCount = 0
    for key,f in stockPriceDic.items():
        dayCount += 1
        # 注意：flag 保留供邻居节点通过图边获取历史信号，但残差路径不引用任何当天字段（见Net.forward）
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
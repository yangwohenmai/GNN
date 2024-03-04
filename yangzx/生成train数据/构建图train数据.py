import tushare as ts
import datetime as dt
import time
import typing
import sys
import os
import baostock as bs
import math
import 获取股市码表
import 获取个股票行情

import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np
from collections import deque
import pandas as pd





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
            stockPriceDic = 获取个股票行情.GetStockPriceDWMBaostock(code, 0)
            if stockPriceDic == False:
                print(code+"行情获取失败")
                continue
            elif len(stockPriceDic) < sampleCount:
                print(code+"低于最小样本限制")
                continue
            else:
                # 根据N来截断数据N=7
                # 构建关系矩阵 1.特征矩阵 2.节点关系矩阵 3.权重矩阵
                dataCount += 1
                print(code+'已输出,序号:NO.'+str(dataCount))
            if dataCount == 500:
                time.sleep(60)
        except Exception as ex:
            print("失败代码："+code+"，异常信息："+str(ex))
    print("finish")
    input()
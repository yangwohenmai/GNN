"""
构建数字化交易状态标签
"""

# <summary>
# 计算策略在时间线上的买卖持仓标签，持仓区间标记1，空仓区间标记0
# </summary>
# <param name="TList">交易日时间线列表</param>
# <param name="buyDateDic">买入时刻表</param>
# <param name="sellDateDic">卖出时刻表</param>
# <param name="BenchmarkList">买入基准</param>
# <param name="CloseList">收盘价</param>
# <returns>持仓周期时间线</returns>
def TimeLineBuyAndSellPeriod1(TList, buyDateDic, sellDateDic):
    resDic = dict()
    flag = -1
    for date in TList:
        if date in buyDateDic:
            flag = 1
            resDic[date] = flag
        elif date in sellDateDic:
            flag = 0
            resDic[date] = flag
        else:
            resDic[date] = flag
    return resDic





# <summary>
# 计算策略在时间线上的买卖持仓标签，持仓区间标记1，空仓区间标记0
# 从buyDateDic和sellDateDic第一次出现值事开始计算，未出现值的日期标记为-1
# </summary>
# <param name="tList">交易日时间线列表</param>
# <param name="buyDateDic">买入时刻表</param>
# <param name="sellDateDic">卖出时刻表</param>
# <param name="BenchmarkList">买入基准</param>
# <param name="CloseList">收盘价</param>
# <returns>持仓周期时间线</returns>
def TimeLineBuyAndSellPeriod(tList, buyDateDic, sellDateDic, longList, shortList):
    resDic = dict()
    flagDic = dict()
    longDic = dict()
    shortDic = dict()
    flag = -1
    for i in range(len(tList)):
        if tList[i] in buyDateDic:
            flag = 1
            flagDic[tList[i]] = flag
            longDic[tList[i]] = longList[i]
            shortDic[tList[i]] = shortList[i]
        elif tList[i] in sellDateDic:
            flag = 0
            flagDic[tList[i]] = flag
            longDic[tList[i]] = longList[i]
            shortDic[tList[i]] = shortList[i]
        else:
            flagDic[tList[i]] = flag
            longDic[tList[i]] = longList[i]
            shortDic[tList[i]] = shortList[i]
    resDic["flagDic"] = flagDic
    resDic["longDic"] = longDic
    resDic["shortDic"] = shortDic
    return resDic
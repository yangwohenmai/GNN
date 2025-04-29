"""
在计算首次买入点时，为了防止产生歧义
有效买卖点数据，从第一次出现完整的金叉开始计算
若首日落在死叉范围，则等待下一个金叉再计算
若首日落在金叉范文，则等待下一个死叉后的金叉再计算
"""
# <summary>
# 简单交叉买入法收益计算
# </summary>
# <param name="tList">时间列表</param>
# <param name="longList">长期线</param>
# <param name="shortList">短期线</param>
# <param name="benchmarkList">买入基准列表，当无法确认真实买入价时，作为暂时买入价格</param>
# <param name="closeList">收盘价</param>
# <returns name="myReturnList">收益率曲线(包含初始价格)</returns>
# <returns name="objectiveRetrunList">客观涨跌价格，即收盘价</returns>
# <returns name="buyDateDic">买点日期</returns>
# <returns name="sellDateDic">卖点日期</returns>
def CrossSimple(tList, longList, shortList, benchmarkList, closeList):
    #昨日交叉状态
    LastStatus = True
    #今日交叉状态
    NowStatus = True
    #首次买入价格
    FirstPrice = 0
    #买入价格
    BuyPrice = -1
    #买入日期
    BuyDate = ''
    #卖出价格
    SellPrice = -1
    #卖出时间
    SellData = ''
    #累计盈利总和
    Money = 0
    #参照首次买入价格，后续每次交易的收益列表
    myReturnList = list()
    #股价自然涨跌列表
    objectiveRetrunList = list()
    # 交易信息
    tradeinfo = ''
    # 买入 日期-价格 列表
    buyDateDic = dict()
    # 卖出 日期-价格 列表
    sellDateDic = dict()
    # 列表
    DataDic = dict()
    if isinstance(tList,list) == False:
        tList = tList.astype(int)
    for i in range(len(tList)):
        #获取SB点,判断短期线上穿长期线
        if longList[i] <= shortList[i]:
            NowStatus = True
        else:
            NowStatus = False
        #如果昨天死叉且今天金叉，则买入，反之卖出
        #由于初始状态是true，true，只有经历过一次死叉后，LastStatus才为false，所以当前if是经历过死叉后的完整金叉，而非残缺金叉，可进行交易。
        if NowStatus == True and LastStatus == False:
            #第一次买入时记录价格，用于计算总收益
            if BuyPrice == -1:
                FirstPrice = closeList[i]
            #如果今天金叉，昨天死叉
            BuyPrice = benchmarkList[i]
            BuyDate = str(tList[i])
            buyDateDic[tList[i]] = BuyPrice
        elif (NowStatus == False and LastStatus == True):
            SellPrice = benchmarkList[i]
            #如果从来没有买过，死叉也不做交易，避免默认LastStatus=true带来的问题
            if BuyPrice != -1:
                Money += SellPrice - BuyPrice
                tradeinfo += "买入：" + str(BuyDate) + "|" + str(BuyPrice) + "---卖出：" + str(tList[i]) + "|" + str(SellPrice) + "---盈利：" + str(Money) + "\r\n"
                sellDateDic[tList[i]] = SellPrice
        else:
            pass
            #(true,true)可能是初始状态，也可能是金叉范围内
            #(false,false)是死叉范围内
            #两种情况均不作处理
        #更新昨天交叉状态
        LastStatus = NowStatus
        # BuyPrice=-1表示从未交易过，历史收益默认为收盘价
        if BuyPrice == -1:
            myReturnList.append(closeList[i])
        else:
            # 后续的总收益建立在第一次买入价格之上
            myReturnList.append(FirstPrice + Money)

        objectiveRetrunList.append(closeList[i])
    
    MyTotalReturn = "收益率：" + str(0 if FirstPrice == 0 else Money / FirstPrice * 100) + "%\r\n"
    #ObjectiveTotalReturn = "自然涨跌：" + str((closeList[len(closeList) - 1] - closeList[0]) / closeList[0] * 100) + "%\r\n"
    ObjectiveTotalReturn = "自然涨跌：" + str((objectiveRetrunList[len(objectiveRetrunList) - 1] - objectiveRetrunList[0]) / objectiveRetrunList[0] * 100) + "%\r\n"
    tradeinfo += MyTotalReturn + ObjectiveTotalReturn
    resultdic = {"tList":tList,"buyDateDic":buyDateDic,"sellDateDic":sellDateDic,"tradeInfo":tradeinfo,"myReturnList":myReturnList,"objectiveRetrunList":objectiveRetrunList,"benchmarkList":benchmarkList,"longList":longList,"shortList":shortList}
    return resultdic



# <summary>
# 计算策略在时间线上的买卖持仓标签
# </summary>
# <param name="TList">交易日时间线列表</param>
# <param name="buyDateDic">买入时刻表</param>
# <param name="sellDateDic">卖出时刻表</param>
# <param name="BenchmarkList">买入基准</param>
# <param name="CloseList">收盘价</param>
# <returns>持仓周期时间线</returns>
def TimeLineBuyAndSellPeriod(TList, buyDateDic, sellDateDic):
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

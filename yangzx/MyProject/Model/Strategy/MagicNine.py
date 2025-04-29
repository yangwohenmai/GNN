import datetime

# 神奇9转指标
def MagicNineFunc(data_close, dateList):
    if len(data_close) <= 13:
        return
    # 取值游标，初始状态前四天作为对比数据
    beginIndex = 4
    winlist = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - 9:
            return winlist
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 向后取9天作为一个单位进行判断
            for j in range(9):
                # 如果单位中每一天价格都小于前四天价格，计数加1，满9则输出
                if data_close[j + i] < data_close[j + i - 4]:
                    downCount += 1
                elif data_close[j + i] > data_close[j + i - 4]:
                    upCount += 1
            # 分析连续9天的状态
            if downCount == 9:
                #print('keep down 9 days:', dateList[i + 8])
                beginIndex = i + 9
                print('({0},{1}) {2}-{3} win:{4}'.format(dateList[i+16],dateList[i+8],data_close[i+16],data_close[i+8],data_close[i+16]-data_close[i+8]))
                win += data_close[i+16]-data_close[i+8]
                winlist.append(win)
            elif upCount == 9:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + 9
            else:
                beginIndex = i
                #print('this unit is not good')
                pass

# 神奇N转（天数可以自定义，默认为9）
def MagicNFunc(data_close, dateList, N = 9):
    #N = 9
    # 默认持仓天数
    holdDay = 9
    # 前四天作为对比数据
    cursor = 4
    # 取值游标
    beginIndex = cursor
    if len(data_close) <= N + cursor:
        return
    winList = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - N:
            return winList
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 每次向未来取N天作为一个单元进行判断
            for j in range(N):
                # 如果单元中某一天价格均小于历史前cursor天价格，下跌天数计数加1
                if data_close[j + i] < data_close[j + i - cursor]:
                    downCount += 1
                # 反之上涨天数计数+1
                elif data_close[j + i] > data_close[j + i - cursor]:
                    upCount += 1
            # 分析连续N天的状态，满9则输出一个买卖点
            if downCount == N:
                #print('keep down N days:', dateList[i + 8])
                beginIndex = i + N
                if (i+N+holdDay)<len(dateList):
                    print('({0},{1}) {2}-{3} win:{4}'.format(dateList[i+N+holdDay],dateList[i+N-1],data_close[i+N+holdDay],data_close[i+N-1],data_close[i+N+holdDay]-data_close[i+N-1]))
                    win += data_close[i+N+holdDay]-data_close[i+N-1]
                else:
                    print('({0},{1}) {2}-{3} win:{4}'.format(dateList[len(dateList)-1],dateList[i+N-1],data_close[len(dateList)-1],data_close[i+N-1],data_close[len(dateList)-1]-data_close[i+N-1]))
                    win += data_close[len(dateList)-1]-data_close[i+N-1]
                winList.append(win)
            elif upCount == N:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + N
            else:
                beginIndex = i
                #print('this unit is not good')
                pass


# 神奇N转标准方法（天数可以自定义，默认为9）
# N:N=9表示九转
# holdDay:默认持仓天数
# winList:收益率曲线(不包含初始价格)
# signalAppearDic:记录日期序列上的买入信号和卖出信号，买信号1，卖信号-1，无信号0
def MagicNFuncStander(data_close, dateList, N = 9, holdDay = 9):
    # 前四天作为对比数据
    cursor = 4
    # 取值游标
    beginIndex = cursor
    # 出现信号日期
    signalAppearDic = dict()
    if len(data_close) <= N + cursor:
        return
    winList = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if dateList[i] not in signalAppearDic:
            signalAppearDic[dateList[i]] = 0
        if i > len(data_close) - N:
            break
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 每次向未来取N天作为一个单元进行判断
            for j in range(N):
                # 如果单元中某一天价格均小于历史前cursor天价格，下跌天数计数加1
                if data_close[j + i] < data_close[j + i - cursor]:
                    downCount += 1
                # 反之上涨天数计数+1
                elif data_close[j + i] > data_close[j + i - cursor]:
                    upCount += 1
            # 分析连续N天的状态，计数满N则输出一个买点或卖点
            if downCount == N:
                beginIndex = i + N
                # 向后取N条超过总数时，则取最后一条
                if (i+N+holdDay)<len(dateList):
                    #print('({0},{1}) {2}-{3} win:{4}'.format(dateList[i+N+holdDay],dateList[i+N-1],data_close[i+N+holdDay],data_close[i+N-1],data_close[i+N+holdDay]-data_close[i+N-1]))
                    win += data_close[i+N+holdDay]-data_close[i+N-1]
                else:
                    #print('({0},{1}) {2}-{3} win:{4}'.format(dateList[len(dateList)-1],dateList[i+N-1],data_close[len(dateList)-1],data_close[i+N-1],data_close[len(dateList)-1]-data_close[i+N-1]))
                    win += data_close[len(dateList)-1]-data_close[i+N-1]
                winList.append(win)
                signalAppearDic[dateList[i+N-1]] = 1
            elif upCount == N:
                beginIndex = i + N
                signalAppearDic[dateList[i+N-1]] = -1
            else:
                beginIndex = i
                #print('this unit is not good')
    return winList, signalAppearDic
                
# 9转+交易区间
def MagicNineFuncNew(data_close, dateList, tradeOption):
    if len(data_close) <= 13:
        return
    # 取值游标，初始状态前四天作为对比数据
    beginIndex = 4
    winlist = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - 9:
            return winlist
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 向后取9天作为一个单位进行判断
            for j in range(9):
                # 如果单位中每一天价格都小于前四天价格，计数加1，满9则输出
                if data_close[j + i] < data_close[j + i - 4]:
                    downCount += 1
                elif data_close[j + i] > data_close[j + i - 4]:
                    upCount += 1
            # 分析连续9天的状态
            if downCount == 9:
                #print('keep down 9 days:', dateList[i + 8])
                beginIndex = i + 9
                #print('({0},{1}) {2}-{3} win:{4}'.format(dateList[i+16],dateList[i+8],data_close[i+16],data_close[i+8],data_close[i+16]-data_close[i+8]))
                #win += data_close[i+16]-data_close[i+8]
                #winlist.append(win)
                
                # 9起始点转落在买周期
                if tradeOption[dateList[i+8]] == 1:
                    # 九转日期
                    date = dateList[i+8]
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    #buyPrice = data_close[i+8]
                    dayCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 买周期内,不做处理
                        if tradeOption[int_date] == 1:
                            pass
                        # 首次死叉，卖出 
                        else:
                            whileFlag = False
                            #win +=  data_close[i+8+dayCount] - data_close[i+8]
                            print('类型1，买入时间',dateList[i+8],'价格',data_close[i+8])
                            print('类型1，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                    #winlist.append(win)
                # 9起始点转落在买区间
                elif tradeOption[dateList[i+8]] == 0:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 起始在卖区间,不做处理
                        if tradeOption[int_date] == 0 and firstBuyFlag == False:
                            pass
                        # 首次进入买区间，买入 
                        elif tradeOption[int_date] == 1 and firstBuyFlag == False:
                            firstBuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型2，买入时间',dateList[i+8],'价格',data_close[i+8+dayCount])
                        # 循环买区间内，不做处理
                        elif tradeOption[int_date] == 1 and firstBuyFlag == True:
                            pass
                        # 二次进入卖区间，卖出
                        elif tradeOption[int_date] == 0 and firstBuyFlag == True:
                            whileFlag = False
                            win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型2，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                    winlist.append(win)
            elif upCount == 9:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + 9
            else:
                beginIndex = i
                #print('this unit is not good')
                pass


# 9转+交易区间（单信号出现后交易两个区间，两次死叉买卖两次）
def MagicNineFuncNew2(data_close, dateList, tradeOption):
    if len(data_close) <= 13:
        return
    # 取值游标，初始状态前四天作为对比数据
    beginIndex = 4
    winlist = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - 9:
            return winlist
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 向后取9天作为一个单位进行判断
            for j in range(9):
                # 如果单位中每一天价格都小于前四天价格，计数加1，满9则输出
                if data_close[j + i] < data_close[j + i - 4]:
                    downCount += 1
                elif data_close[j + i] > data_close[j + i - 4]:
                    upCount += 1
            # 分析连续9天的状态
            if downCount == 9:
                #print('keep down 9 days:', dateList[i + 8])
                beginIndex = i + 9
                # 9转起始点落在买周期
                if tradeOption[dateList[i+8]] == 1:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    BuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        int_date = int(date)
                        if int_date not in tradeOption:
                            date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                            continue
                        # 起始在买区间,直接买入
                        if tradeOption[int_date] == 1 and BuyFlag == False:
                            BuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型1，买入时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 首次进入卖区间，卖出 
                        elif tradeOption[int_date] == 0 and BuyFlag == True:
                            intervalCount += 1
                            BuyFlag = False
                            win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型1，卖出时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 循环在买区间内，不做处理
                        elif tradeOption[int_date] == 1 and BuyFlag == True:
                            pass
                        # 循环在卖区间，不做处理
                        elif tradeOption[int_date] == 0 and BuyFlag == False:
                            pass
                        if intervalCount == 2:
                            whileFlag = False
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        dayCount += 1
                    winlist.append(win)
                # 9起始点转落在卖区间
                elif tradeOption[dateList[i+8]] == 0:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 起始在卖区间,不做处理
                        if tradeOption[int_date] == 0 and firstBuyFlag == False:
                            pass
                        # 首次进入买区间，买入 
                        elif tradeOption[int_date] == 1 and firstBuyFlag == False:
                            firstBuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型2，买入时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 循环买区间内，不做处理
                        elif tradeOption[int_date] == 1 and firstBuyFlag == True:
                            pass
                        # 二次进入卖区间，卖出
                        elif tradeOption[int_date] == 0 and firstBuyFlag == True:
                            firstBuyFlag = False
                            intervalCount += 1
                            #win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型2，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                        if intervalCount == 2:
                            whileFlag = False
                    #winlist.append(win)
            elif upCount == 9:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + 9
            else:
                beginIndex = i
                #print('this unit is not good')
                pass




# 9转+交易区间（单信号出现后交易两个区间，两次死叉买卖两次）
def MagicNineFuncNew3(data_close, dateList, tradeOption):
    if len(data_close) <= 13:
        return
    # 取值游标，初始状态前四天作为对比数据
    beginIndex = 4
    winlist = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - 9:
            return winlist
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 向后取9天作为一个单位进行判断
            for j in range(9):
                # 如果单位中每一天价格都小于前四天价格，计数加1，满9则输出
                if data_close[j + i] < data_close[j + i - 4]:
                    downCount += 1
                elif data_close[j + i] > data_close[j + i - 4]:
                    upCount += 1
            # 分析连续9天的状态
            if downCount == 9:
                #print('keep down 9 days:', dateList[i + 8])
                beginIndex = i + 9
                # 9转起始点落在买周期
                if tradeOption[dateList[i+8]] == 1:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    BuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        int_date = int(date)
                        if int_date not in tradeOption:
                            date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                            continue
                        # 起始在买区间,直接买入
                        if tradeOption[int_date] == 1 and BuyFlag == False:
                            BuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型1，买入时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 首次进入卖区间，卖出 
                        elif tradeOption[int_date] == 0 and BuyFlag == True:
                            intervalCount += 1
                            BuyFlag = False
                            win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型1，卖出时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 循环在买区间内，不做处理
                        elif tradeOption[int_date] == 1 and BuyFlag == True:
                            pass
                        # 循环在卖区间，不做处理
                        elif tradeOption[int_date] == 0 and BuyFlag == False:
                            pass
                        if intervalCount == 2:
                            whileFlag = False
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        dayCount += 1
                    winlist.append(win)
                # 9起始点转落在卖区间
                elif tradeOption[dateList[i+8]] == 0:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 起始在卖区间,不做处理
                        if tradeOption[int_date] == 0 and firstBuyFlag == False:
                            pass
                        # 首次进入买区间，买入 
                        elif tradeOption[int_date] == 1 and firstBuyFlag == False:
                            firstBuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型2，买入时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 循环买区间内，不做处理
                        elif tradeOption[int_date] == 1 and firstBuyFlag == True:
                            pass
                        # 二次进入卖区间，卖出
                        elif tradeOption[int_date] == 0 and firstBuyFlag == True:
                            firstBuyFlag = False
                            intervalCount += 1
                            win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型2，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                        if intervalCount == 2:
                            whileFlag = False
                    winlist.append(win)
            elif upCount == 9:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + 9
            else:
                beginIndex = i
                #print('this unit is not good')
                pass





# 9转+交易区间（单信号出现后交易两个区间,两次死叉累计买卖一次）
def MagicNineFuncNew4(data_close, dateList, tradeOption):
    if len(data_close) <= 13:
        return
    # 取值游标，初始状态前四天作为对比数据
    beginIndex = 4
    winlist = list()
    # 累计收益
    win = 0
    for i in range(len(data_close)):
        if i > len(data_close) - 9:
            return winlist
        if i >= beginIndex:
            # 连续下跌天数
            downCount = 0
            # 连续上涨天数
            upCount = 0
            # 向后取9天作为一个单位进行判断
            for j in range(9):
                # 如果单位中每一天价格都小于前四天价格，计数加1，满9则输出
                if data_close[j + i] < data_close[j + i - 4]:
                    downCount += 1
                elif data_close[j + i] > data_close[j + i - 4]:
                    upCount += 1
            # 分析连续9天的状态
            if downCount == 9:
                #print('keep down 9 days:', dateList[i + 8])
                beginIndex = i + 9
                
                # 9转起始点落在买周期
                if tradeOption[dateList[i+8]] == 1:
                    # 九转日期
                    date = dateList[i+8]
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    #buyPrice = data_close[i+8]
                    # 昨日周期
                    yesterday = True
                    # 今日周期
                    today = True
                    # 向后推算天数
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 买周期内,不做处理
                        if tradeOption[int_date] == 1:
                            today = True
                            pass
                        # 卖周期标记
                        else:
                            today = False
                            #whileFlag = False
                            #win +=  data_close[i+8+dayCount] - data_close[i+8]
                            #print('类型1，买入时间',dateList[i+8],'价格',data_close[i+8])
                            #print('类型1，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                        # 如果昨买今卖，记为一次死叉
                        if yesterday == True and today == False:
                            intervalCount += 1
                        # 两次死叉即可卖出
                        if intervalCount == 2:
                            whileFlag = False
                            win +=  data_close[i+8+dayCount] - data_close[i+8]
                            print('类型1，买入时间',dateList[i+8],'价格',data_close[i+8])
                            print('类型1，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                        yesterday = today
                    winlist.append(win)
                # 9起始点转落在卖区间
                elif tradeOption[dateList[i+8]] == 0:
                    # 九转日期
                    date = dateList[i+8]
                    # 是否买入
                    firstBuyFlag = False
                    whileFlag = True
                    # 买入价
                    buyPrice = 0
                    dayCount = 0
                    # 交易区间个数统计
                    intervalCount = 0
                    while(whileFlag):
                        date = (datetime.datetime.strptime(str(date), '%Y%m%d')+ datetime.timedelta(days=1)).strftime('%Y%m%d')
                        int_date = int(date)
                        if int_date not in tradeOption:
                            continue
                        dayCount += 1
                        # 起始在卖区间,不做处理
                        if tradeOption[int_date] == 0 and firstBuyFlag == False:
                            pass
                        # 首次进入买区间，买入 
                        elif tradeOption[int_date] == 1 and firstBuyFlag == False:
                            firstBuyFlag = True
                            buyPrice = data_close[i+8+dayCount]
                            print('类型2，买入时间',dateList[i+8+dayCount],'价格',data_close[i+8+dayCount])
                        # 循环买区间内，不做处理
                        elif tradeOption[int_date] == 1 and firstBuyFlag == True:
                            pass
                        # 二次进入卖区间，卖出
                        elif tradeOption[int_date] == 0 and firstBuyFlag == True:
                            firstBuyFlag = False
                            intervalCount += 1
                            win +=  data_close[i+8+dayCount] - buyPrice
                            print('类型2，卖出时间',int_date,'价格',data_close[i+8+dayCount])
                        if intervalCount == 2:
                            whileFlag = False
                    winlist.append(win)
            elif upCount == 9:
                #print('keep up 9 days:', dateList[i + 8])
                beginIndex = i + 9
            else:
                beginIndex = i
                #print('this unit is not good')
                pass









if __name__ == '__main__':
    closeList = [26.56,25.04,
                    23.10,23.14,22.70,22.39,22.89,24.17,25.30,25.80,26.86,27.47,28.44,28.67,27.64,27.84,28.03,27.55,
                    26.67,25.37,25.39,26.66,26.39,25.85,24.40,25.13,24.38,24.15,24.19,23.88,22.66,24.00,23.38,24.86,
                    27.14,27.78,28.79,29.66,27.52,26.60,26.38,25.19,24.93,25.87,24.99,23.95,23.90,24.80,25.98,25.80,
                    24.16,24.56,23.58,23.98,24.00,23.43,23.93,24.18,23.93,23.92,23.70,23.75,25.75,25.30,25.58,26.60,
                    26.37,24.19,24.08,23.95,24.21,23.46,23.84,24.21,23.59,23.52,23.11,22.11,20.88,19.85,19.38,18.28,
                    18.00,18.01,17.77,18.16,17.75,18.06,18.08,18.27,17.77,17.88,17.94,18.09,19.90,19.05,19.10,20.76,
                    21.38,19.73,20.58,20.11,19.76,19.23,19.49,18.85,18.73,18.10,17.77,18.12,18.44,18.51,18.65,19.24,
                    18.42,18.36,17.99,17.87,18.03,18.15]
    dateList = [210104,210105,210106,210107,210108,210111,210112,210113,210114,210115,210118,210119,210120,210121,210122,210125,
                210126,210127,210128,210129,210201,210202,210203,210204,210205,210208,210209,210210,210218,210219,210222,210223,
                210224,210225,210226,210301,210302,210303,210304,210305,210308,210309,210310,210311,210312,210315,210316,210317,
                210318,210319,210322,210323,210324,210325,210326,210329,210330,210331,210401,210402,210406,210407,210408,210409,
                210412,210413,210414,210415,210416,210419,210420,210421,210422,210423,210426,210427,210428,210429,210430,210506,
                210507,210510,210511,210512,210513,210514,210517,210518,210519,210520,210521,210524,210525,210526,210527,210528,
                210531,210601,210602,210603,210604,210607,210608,210609,210610,210611,210615,210616,210617,210618,210621,210622,
                210623,210624,210625,210628,210629,210630,210701,210702]

    #MagicNineFuncNew(closeList2, dateList)





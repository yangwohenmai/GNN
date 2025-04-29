# 单一信号+区间叠加交易法
# signalDateList：单一信号买点日期列表
# dateList：法定交易日列表
# closeList：所有交易日的收盘价
# tradeOption：叠加买卖区间
def SignalIntervalTradeFunc(closeList, dateList, signalDateList, tradeOption):
    # 遍历所有交易日
    for i in range(len(dateList)):
        # 如果当天出现单一交易信号，开始计算叠加区间
        if dateList[i] in signalDateList:
            # 若起始点落在买区间
            if tradeOption[dateList[i]] == 1:
                firstBuyFlag = False
                whileFlag = True
                date = dateList[i]
                # 买入价
                buyPrice = closeList[i]
                # 买区间天数
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
                        win +=  closeList[i+dayCount] - closeList[i]
                        print('类型1，买入时间',dateList[i],'价格',closeList[i])
                        print('类型1，卖出时间',int_date,'价格',closeList[i+dayCount])
                winlist.append(win)
            # 单信号起始点落在卖区间
            elif tradeOption[dateList[i]] == 0:
                # 单信号日期
                date = dateList[i]
                # 是否首次买入
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
                    # 起始在卖区间,卖区间内不做处理
                    if tradeOption[int_date] == 0 and firstBuyFlag == False:
                        pass
                    # 首次进入买区间，买入 
                    elif tradeOption[int_date] == 1 and firstBuyFlag == False:
                        firstBuyFlag = True
                        buyPrice = closeList[i + dayCount]
                        print('类型2，买入时间',dateList[i],'价格',closeList[i + dayCount])
                    # 循环买区间内，不做处理
                    elif tradeOption[int_date] == 1 and firstBuyFlag == True:
                        pass
                    # 二次进入卖区间，卖出
                    elif tradeOption[int_date] == 0 and firstBuyFlag == True:
                        win +=  closeList[i + dayCount] - buyPrice
                        whileFlag = False
                        print('类型2，卖出时间',int_date,'价格',closeList[i + dayCount])
                winlist.append(win)
    return winlist
                
                
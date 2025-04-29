import typing
from Strategy.EMA import EMA
from Strategy.REF import REF
import tushare as ts
#/// <summary>
#/// 捕捞季节指标
#/// VAR1:=(2*CLOSE+HIGH+LOW)/3;
#/// VAR2:= EMA(EMA(EMA(VAR1, 3), 3), 3);
#/// J: (VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100;
#/// D: MA(J, 2);
#/// K: MA(J, 1);
#/// </summary>
#/// <param name="stocklist"></param>
#/// <param name="period">计算天数，因为有递归计算，所以设置period值尽量略小于len(stockDic)为佳，否则样本点不足</param>
#/// <returns></returns>
def BLJJ(stockDic, period = 30):
    testValues = list()
    #时间列表
    tList = list()
    #收盘价列表
    closeList = list()
    #开盘价列表
    openList = list()
    #最高价列表
    highList = list()
    #最低价列表
    lowList = list()
    #买入基准价
    arg = list()
    #(2*CLOSE+HIGH+LOW)/3
    for key,value in stockDic.items():
        testValues.append((2 * value["close"] + value["high"] + value["low"]) / 3)
        #testValues.Add((4 * item.Value.close) / 3);
        #testValues.Add((4 * item.Value.open) / 3);

    #J: (VAR2 - REF(VAR2, 1)) / REF(VAR2, 1) * 100
    pre_ema_list = EMA(EMA(EMA(testValues, 3)["Values"], 3)["Values"], 3)["Values"]
    DList = list()
    KList = list()

    #根据给定显示周期period，逆时序截取period条数据：D、K线，Time，开盘价，收盘价，最高价，最低价
    if len(pre_ema_list) > period:
        for i in range(period):
            J1 = (REF(pre_ema_list, i) - REF(pre_ema_list, i + 1)) / REF(pre_ema_list, i + 1) * 100
            J2 = (REF(pre_ema_list, i + 1) - REF(pre_ema_list, i + 2)) / REF(pre_ema_list, i + 2) * 100
            DList.append((J1 + J2) / 2)
            KList.append(J1)
            tList.append((list(stockDic.keys())[len(stockDic) - i - 1]))
            closeList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["close"])
            openList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["open"])
            highList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["high"])
            lowList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["low"])
    else:
        for i in range(len(pre_ema_list)-1):
            J1 = (REF(pre_ema_list, i) - REF(pre_ema_list, i + 1)) / REF(pre_ema_list, i + 1) * 100
            J2 = (REF(pre_ema_list, i + 1) - REF(pre_ema_list, i + 2)) / REF(pre_ema_list, i + 2) * 100
            DList.append(J1 + J2)
            KList.append(J1)
            tList.append((list(stockDic.keys())[len(stockDic) - i - 1]))
            closeList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["close"])
            openList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["open"])
            highList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["high"])
            lowList.append(stockDic[(list(stockDic.keys())[len(stockDic) - i - 1])]["low"])
    DList.reverse()
    KList.reverse()
    tList.reverse()
    closeList.reverse()
    openList.reverse()
    highList.reverse()
    lowList.reverse()

    # 默认基准价取开盘价和收盘价的中值，若不取中值，则要将买卖价格做对称处理，分别计算。
    # eg:距高点1/3买入，距低点1/3卖出。(即高买低卖更符合真实交易场景)
    for i in range(len(closeList)):
        arg.append((closeList[i] + openList[i]) / 2)#中值
        #arg.append(closeList[i]*2/3 + openList[i]/3)#1/3值

    result = {"StockCode":"null", "longList":DList, "shortList":KList, "tList":tList, "closeList":closeList, "openList":openList, "highList":highList, "lowList":lowList, "benchmarkList":arg }
    #clo = "H,O,L,C,Long,Short\n"
    #for i in range(len(result["tList"])):
    #    clo += "'%s','%s','%s','%s','%s','%s'\n"%(result["highList"][i], result["openList"][i], result["lowList"][i], result["closeList"][i], result["longList"][i], result["shortList"][i])
    #print(clo)
    return result


# stock对象
class Stock:
    def __init__(self,open,close,low,high):
        #self.open = open
        #self.close = close
        #self.low = low
        #self.high = high
        pass
    
    open = 0
    close = 0
    low = 0
    high = 0
    tdate = 19000101
    
    def say(self, content):
        print(content)

# 获取完整股票列表
def SaveAllStockList():
    pro = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol')
    allstocklistdic = dict()
    allstocklistdic1 = dict()
    for col in df.values:
        allstocklistdic[col[0]] = col[1]
    for col in df.values:
        allstocklistdic1[col[1]] = col[0]
    print('AllStockList has save')
    return allstocklistdic, allstocklistdic1


if __name__ == '__main__':
    pro = ts.set_token('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    pro1 = ts.pro_api('6d9ac99d25b0157dcbb1ee3d35ef1250e5295ff80bb59741e1a56b35')
    # 获取股票码表
    allstocklistdic,allstocklistdic1 = SaveAllStockList()
    code = allstocklistdic1["300083"]
    df = ts.pro_bar(ts_code=code, start_date="20211011", end_date="20220120")
    stockList = list()
    dic = dict()
    stockDic = dict()
    stockInfo = Stock
    for i in range(len(df.values)):
        value = df.values[i]
        param = {'symbol':value[0], 'tdate':value[1], 'open':value[2], 'high':value[3], \
            'low':value[4], 'close':value[5], 'lclose':value[6], 'vol':value[9], 'amount':value[10]}
        dic = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5]}
        stockInfo.tdate = value[1]
        stockInfo.open = value[2]
        stockInfo.high = value[3]
        stockInfo.low = value[4]
        stockInfo.close = value[5]
        stockList.append(dic)
    OrderDic = typing.OrderedDict()
    stockList.reverse()
    for item in stockList:
        OrderDic[item['tdate']] = item
    BLJJ(OrderDic,60)
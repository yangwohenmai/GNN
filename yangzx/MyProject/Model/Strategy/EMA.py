import numpy as np
# <summary>
# 移动平均指数EMA
# </summary>
# <param name="inputList">Input signal</param>
# <param name="period">Number of periods</param>
# <returns>Object containing operation results</returns>
def EMA(inputList, period):
    returnValues = list()
    multiplier = 2.0 / (period + 1)
    initialSMA = np.mean(inputList[0 : period])
    returnValues.append(initialSMA)
    copyInputValues = inputList

    for i in range(len(copyInputValues)):
        if i < period:
            continue
        resultValue = (copyInputValues[i] - returnValues[-1]) * multiplier + returnValues[-1]
        returnValues.append(resultValue)
    
    returnDic = dict()
    returnDic["Values"] = returnValues
    returnDic["StartIndexOffset"] = period - 1

    return returnDic
    
    
if __name__ == '__main__':
    pass
    
    
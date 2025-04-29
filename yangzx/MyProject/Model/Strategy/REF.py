# 取前N个周期数据
# inputList:输入的数据列表
# 向前取多少个周期的数据
def REF(inputList, period):
    return inputList[len(inputList) - period - 1]
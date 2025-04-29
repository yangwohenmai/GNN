import math
from statistics import mean
import numpy
import pandas as pd
from pandas import Series
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
"""
数据特征工程类库
差分/逆差分
归一化/逆归一化
标准化/逆标准化
剔除极值
"""
def test():
    testPredictPlot = numpy.empty(10)
    print(testPredictPlot)
    #testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    print(testPredictPlot)

    # 加载数据
    dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # 缩放数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # 对数据进行逆缩放
    dataset = scaler.inverse_transform(dataset)

# 差分转换，差分后序列首值被抛弃
# interval转换步长
# dataset待转换序列
#def difference1(dataset, interval=1):
#    if isinstance(dataset, pd.DataFrame) == False:
#        dataset = pd.DataFrame(dataset)
#    firstItem = dataset[0]
#    datasetnew = dataset.shift(axis=0, periods=1)
#    datasetnew = datasetnew.fillna(0, inplace=False)
#    dataset = dataset - datasetnew
#    dataset = dataset.shift(axis=0, periods=-1)
#    dataset = dataset.dropna(axis=0,how='all')
#    print(dataset)
#    return dataset, firstItem

# 差分，差分后序列首值被抛弃
# interval转换步长
# dataset待转换序列
#def difference(dataset, interval=1):
#    diff = list()
#    firstItem = dataset[0]
#    for i in range(interval, len(dataset)):
#        value = dataset[i] - dataset[i - interval]
#        diff.append(value)
#    return Series(diff), firstItem

# 逆差分，传入原序列首值，反求出后续值
# first_item原序列首值
# 待转换序列
#def inverse_difference(first_item, dataset):
#    if isinstance(dataset, pd.DataFrame) == False:
#        dataset = pd.DataFrame(dataset)
#    dataset = dataset.shift(axis=0, periods=1)
#    dataset = dataset.fillna(0, inplace=False)
#    print(dataset)
#    return dataset

# 多维数组，每列分别归一化
#def MinMaxScalerTransform2D(list):
#    if isinstance(list, pd.DataFrame) == False:
#        list = pd.DataFrame(list)
#    dataset = list
#    mins = dataset.min(0)
#    maxs = dataset.max(0) #返回data矩阵中每一列中最大的元素，返回一个列表
#    ranges1 = maxs - mins
#    normData = numpy.zeros(numpy.shape(dataset))
#    row = dataset.shape[0]
#    normData = dataset - numpy.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
#    normData = normData / numpy.tile(ranges1,(row,1)) #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
#    return normData

"""
多维数组差分转换，差分后序列首行/列被抛弃
interval 差分步长
dataset 待差分序列
axispara 选择行/列差分,0为行间差分，1为列间差分
"""
def DifferenceFunc(dataset, axispara=0, interval=1):
    if isinstance(dataset, pd.DataFrame) == False:
        dataset = pd.DataFrame(dataset)
    # 获取首行/列，作为参数传出
    if axispara == 0:
        firstItem = dataset.loc[0,:]
    else:
        firstItem = dataset.loc[:,0]
    # 逐行/列差分
    dataset = dataset.diff(periods=interval, axis=axispara)
    # 删除差分后的首行/列nan值
    dataset = dataset.dropna(axis=axispara, how='all').reset_index(drop=True)
    return dataset.values, Series(firstItem)

"""
多维数组逆差分转化，传入原序列首行/列，反求出后续值
first_item 原序列首行/列
dataset 待差分序列
axispara 选择行/列差分,0为行间差分，1为列间差分
"""
def DifferenceInverseFunc(first_item, dataset, axispara=0):
    if isinstance(dataset, pd.DataFrame) == False:
        dataset = pd.DataFrame(dataset)
    # 将首行/列
    if axispara == 0:
        first_item = pd.DataFrame(first_item).T
    else:
        first_item = pd.DataFrame(first_item)
    # 将首行/列与差分序列合并
    dataset = pd.concat([first_item,dataset], axis=axispara, ignore_index=True)
    # 逐行/列逆差分
    for i in range(dataset.shape[axispara]):
        if i > 0 and axispara == 0:
            dataset.loc[i,:] += dataset.loc[i-1,:]
        elif i > 0:
            dataset.loc[:,i] += dataset.loc[:,i-1]
    return dataset.values

"""
对构建好的监督学习数据进行差分
将数据集中每一个特征列取出来，分别进行差分，再将差分后的特征列数据重新替换回去
supervisedLearningData 构建好的监督学习数据X->(sample,timesteps,feature)->[[[f1,f2],[f1,f2],...],[[f1,f2],[f1,f2],...],...]
datay 构建好的监督学习数据Y，仅用于按照n阶差分剔除前n行无效数据
diffTime 差分阶数，进行n阶差分后，最终返回值要剔除前n行无效数据
include 仅对传入数组中数值所对应的特征列做差分，从0开始算，exclude=[1,4]表示第2,5列特征进行差分，空数组表示对所有列不差分
此方法尚未写对应逆差分，暂不可用于数值预测
"""
def DiffSupervisedFunc(supervisedLearningData, datay, include=[], diffTime=1):
    # 每次循环，将监督学习数据中的一个特征取列出来，用于后续进行差分
    for n in range(numpy.array(supervisedLearningData).shape[2]):
        if n not in include:
            continue
        featureList = list()
        # 将当前第n个特征列的特征全部取出存入featureList，n表示有n个特征
        for i in range(len(supervisedLearningData)):
            if i == len(supervisedLearningData)-1:
                # 如果是最后一个sample，取这个sample中所有行的，第n列特征
                for item in supervisedLearningData[i]:
                    featureList.append(item[n])
            else:
                # 取第i个sample中第0行的，第n列的特征值
                featureList.append(supervisedLearningData[i][0][n])

        # 对当前取出的第n个特征列进行差分，存入diffFeatureList
        diffFeatureList = numpy.diff(featureList, diffTime).tolist()
        # 差分后剔除的数据补0，方便后续替换数据
        for index in range(diffTime):
            diffFeatureList.insert(index, 0)

        # 将当前第n个特征列差分后的数据diffFeatureList，替换给原监督学习数据集supervisedLearningData对应的特征列上
        for i in range(len(supervisedLearningData)):
            for j in range(len(supervisedLearningData[i])):
                supervisedLearningData[i][j][n] = diffFeatureList[i+j]
    # 剔除差分后产生的无效行数据
    return supervisedLearningData[diffTime:], datay[diffTime:]

"""
多维数组归一化，对每列分别归一化,回传最大最小值差序列、和最小值序列用于后续逆归一化
axispara选择行/列归一化,0为列间归一化，1为行间归一化
"""
def MinMaxScalerFunc(dataset, axispara=0):
    if isinstance(dataset, pd.DataFrame) == False:
        dataset = pd.DataFrame(dataset)
    # 获取每一行(axispara=1)/列(axispara=0)的最大最小值
    maxList,minList=dataset.max(axis=axispara),dataset.min(axis=axispara)
    # 获取每一行(axispara=1)/列(axispara=0)的最大最小值之差
    diffList = maxList-minList
    # 对每行(axispara=1)/列(axispara=0)进行归一化计算
    if axispara == 0:
        dataset = (dataset-minList)/diffList
    else:
        dataset = (dataset.T-minList)/diffList
        dataset = dataset.T
    return dataset.values, diffList, minList

"""
多维数组逆归一化，对每列分别逆归一化
diffList 最大最小值差序列
minList 最小值序列
axispara 选择行/列归一化,0为列间归一化，1为行间归一化
"""
def MinMaxScalerInverseFunc(dataset, diffList, minList, axispara=0):
    if isinstance(dataset, pd.DataFrame) == False:
        dataset = pd.DataFrame(dataset)
    if(axispara == 0):
        # 列间逆归一化
        dataset = dataset * diffList + minList
        #dataset = dataset * diffList.T + minList
    else:
        # 行间逆归一化
        dataset = (dataset.T * diffList.T + minList.T).T
    return dataset.values
    
"""
对构建好的监督学习数据进行最值归一化->[0, 1]
将数据集中每一个特征列取出来，分别进行归一化，再将归一化后的特征列数据重新替换回去
supervisedLearningData 构建好的监督学习数据->(sample,timesteps,feature)->[[[f1,f2],[f1,f2],...],[[f1,f2],[f1,f2],...],...]
exclude 排除传入数组中数值所对应的特征列不做归一化，从0开始算，exclude=[1,4]表示第2,5列特征不进行归一化，空数组表示对所有列进行归一化
此方法尚未写对应逆归一化，暂不可用于数值预测
"""
def MinMaxSupervisedFunc(supervisedLearningData, exclude=[]):
    # 每次循环，将监督学习数据中的一个特征取列出来，用于后续进行归一化
    for n in range(numpy.array(supervisedLearningData).shape[2]):
        if n in exclude:
            continue
        featureList = list()
        # 将当前第n个特征列的特征全部取出存入featureList，n表示有n个特征
        for i in range(len(supervisedLearningData)):
            if i == len(supervisedLearningData)-1:
                # 如果是最后一个sample，取这个sample中所有行的，第n列特征
                for item in supervisedLearningData[i]:
                    featureList.append(item[n])
            else:
                # 取第i个sample中第0行的，第n列的特征值
                featureList.append(supervisedLearningData[i][0][n])

        # 对当前取出的第n个特征列进行归一化，存入normalizationFeatureList
        diff = max(featureList) - min(featureList)
        normalizationFeatureList = (numpy.array(featureList) - min(featureList))/diff
        normalizationFeatureList = normalizationFeatureList.tolist()

        # 将当前第n个特征列归一化后的数据normalizationFeatureList，替换给原监督学习数据集supervisedLearningData对应的特征列上
        for i in range(len(supervisedLearningData)):
            for j in range(len(supervisedLearningData[i])):
                supervisedLearningData[i][j][n] = normalizationFeatureList[i+j]

    return supervisedLearningData

"""
对构建好的监督学习数据进行均值归一化->[-1, 1]
将数据集中每一个特征列取出来，分别进行均值归一化，再将均值归一化后的特征列数据重新替换回去
supervisedLearningData 构建好的监督学习数据->(sample,timesteps,feature)->[[[f1,f2],[f1,f2],...],[[f1,f2],[f1,f2],...],...]
exclude 排除传入数组中数值所对应的特征列不做均值归一化，从0开始算，exclude=[1,4]表示第2,5列特征不进行均值归一化，空数组表示对所有列进行均值归一化
此方法尚未写对应逆均值归一化，暂不可用于数值预测
"""
def MeanSupervisedFunc(supervisedLearningData, exclude=[]):
    # 每次循环，将监督学习数据中的一个特征取列出来，用于后续进行均值归一化
    for n in range(numpy.array(supervisedLearningData).shape[2]):
        if n in exclude:
            continue
        featureList = list()
        # 将当前第n个特征列的特征全部取出存入featureList，n表示有n个特征
        for i in range(len(supervisedLearningData)):
            if i == len(supervisedLearningData)-1:
                # 如果是最后一个sample，取这个sample中所有行的，第n列特征
                for item in supervisedLearningData[i]:
                    featureList.append(item[n])
            else:
                # 取第i个sample中第0行的，第n列的特征值
                featureList.append(supervisedLearningData[i][0][n])

        # 对当前取出的第n个特征列进行均值归一化，存入meanNormalizationFeatureList
        diff = max(featureList) - min(featureList)
        meanNormalizationFeatureList = (numpy.array(featureList) - mean(featureList))/diff
        meanNormalizationFeatureList = meanNormalizationFeatureList.tolist()

        # 将当前第n个特征列均值归一化后的数据meanNormalizationFeatureList，替换给原监督学习数据集supervisedLearningData对应的特征列上
        for i in range(len(supervisedLearningData)):
            for j in range(len(supervisedLearningData[i])):
                supervisedLearningData[i][j][n] = meanNormalizationFeatureList[i+j]

    return supervisedLearningData

"""
二维数组归一化，所有数据按照全局标准归一化
"""
def MinMaxScalerTransformAll(list):
    if isinstance(list, pd.DataFrame) == False:
        list = numpy.array(list)
    dataset = list
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return dataset

"""
多维数组标准化
axispara选择行/列标准化,0为列间标准化，1为行间标准化
"""
def ZeroStandardFunc(dataset, axispara=0):
    # 一维序列重构为二维
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1, 1) 
        #dataset = dataset.reshape(len(dataset), 1)
    # 取每行/列的均值
    average = numpy.mean(dataset,axis=axispara)
    # 取每行/列的标准差
    stddev = numpy.std(dataset,axis=axispara)
    # (X-average)/stddev
    if axispara == 1:
        numerator = dataset - average.reshape(dataset.shape[0],1).repeat(dataset.shape[1],axis=axispara)
        dataset = numerator / stddev.reshape(dataset.shape[0],1).repeat(dataset.shape[1],axis=axispara)
    else:
        numerator = dataset - average.reshape(1,dataset.shape[1]).repeat(dataset.shape[0],axis=axispara)
        dataset = numerator / stddev.reshape(1,dataset.shape[1]).repeat(dataset.shape[0],axis=axispara)
    return dataset,average,stddev

"""
多维数组逆标准化
axispara选择行/列逆标准化,0为列间逆标准化，1为行间逆标准化
"""
def ZeroStandardInverseFunc(dataset, average, stddev, axispara=0):
    # 一维序列重构为二维
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1, 1) 
    if axispara == 1:
        dataset = dataset * stddev.reshape(dataset.shape[0],1).repeat(dataset.shape[1],axis=axispara)
        dataset = dataset + average.reshape(dataset.shape[0],1).repeat(dataset.shape[1],axis=axispara)
    else:
        dataset = dataset * stddev.reshape(1,dataset.shape[1]).repeat(dataset.shape[0],axis=axispara)
        dataset = dataset + average.reshape(1,dataset.shape[1]).repeat(dataset.shape[0],axis=axispara)
    return dataset

"""
对构建好的监督学习数据进行标准化
将数据集中每一个特征列取出来，分别进行标准化，再将标准化后的特征列数据重新替换回去
supervisedLearningData 构建好的监督学习数据->(sample,timesteps,feature)->[[[f1,f2],[f1,f2],...],[[f1,f2],[f1,f2],...],...]
exclude 排除传入数组中数值所对应的特征列不做标准化，从0开始算，exclude=[1,4]表示第2,5列特征不进行标准化，空数组表示对所有列进行标准化
此方法尚未写对应逆标准化，暂不可用于数值预测
"""
def StandardizationSupervisedFunc(supervisedLearningData, exclude=[]):
    # 每次循环，将监督学习数据中的一个特征取列出来，用于后续进行标准化
    for n in range(numpy.array(supervisedLearningData).shape[2]):
        if n in exclude:
            continue
        featureList = list()
        # 将当前第n个特征列的特征全部取出存入featureList，n表示有n个特征
        for i in range(len(supervisedLearningData)):
            if i == len(supervisedLearningData)-1:
                # 如果是最后一个sample，取这个sample中所有行的，第n列特征
                for item in supervisedLearningData[i]:
                    featureList.append(item[n])
            else:
                # 取第i个sample中第0行的，第n列的特征值
                featureList.append(supervisedLearningData[i][0][n])
        
        # 对当前取出的第n个特征列进行标准化，存入standardizationFeatureList->(X-average)/stddev
        featureListAverage = numpy.mean(featureList)
        featureListStddev = numpy.std(featureList)
        standardizationFeatureList = (numpy.array(featureList) - featureListAverage) / featureListStddev
        standardizationFeatureList = standardizationFeatureList.tolist()

        # 将当前第n个特征列标准化后的数据standardizationFeatureList，替换给原监督学习数据集supervisedLearningData对应的特征列上
        for i in range(len(supervisedLearningData)):
            for j in range(len(supervisedLearningData[i])):
                supervisedLearningData[i][j][n] = standardizationFeatureList[i+j]

    return supervisedLearningData

"""
多维数组剔除极值
axispara选择行/列剔除极值,0为行间操作，1为列间操作
"""
def ExtremePoint(dataset, axispara=1):
    if isinstance(dataset, pd.DataFrame) == False:
        dataset = pd.DataFrame(dataset)
    values = dataset.values
    if axispara == 0:
        for i in range(values.shape[axispara]):
            # 计算每行数据的均值和标准差
            data_mean, data_std = numpy.mean(values[i,:]), numpy.std(values[i,:])
            # 定义异常值的上界和下界为（均值+3倍标准差）
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # 将极值替换成上下限值
            values[i,:][values[i,:] > upper] = upper
            values[i,:][values[i,:] < lower] = lower
    else:
        for i in range(values.shape[axispara]):
            # 计算每列数据的均值和标准差
            data_mean, data_std = numpy.mean(values[:,i]), numpy.std(values[:,i])
            # 定义异常值的上界和下界为（均值+3倍标准差）
            cut_off = data_std * 3
            lower, upper = data_mean - cut_off, data_mean + cut_off
            # 将极值替换成上下限值
            values[:,i][values[:,i] > upper] = upper
            values[:,i][values[:,i] < lower] = lower
    print(values)
    return values




if __name__ == '__main__':
    list1 = []
    for i in range(10):
        list1.append(i*i)
    list2 = numpy.zeros((5,5))
    for i in range(5):
        for j in range(5):
            list2[i][j] = (j+1)*(i+1)
    list3=numpy.random.randint(1, 10, size=(15, 15)).astype('float32')
    data1 = numpy.array([1,2,3,1000,4,5,6,7,8,9,10,11,12,13,14,-800,16])
    # shape(2,3,4)
    list4 = [[[1,2,3,4],[2,3,4,5],[3,4,5,6]],[[2,3,4,5],[3,4,5,6],[4,5,6,7]]]
    
    # 剔除极值
    list3[2][1] = 1000
    print(ExtremePoint(data1))

    
    # 标准化
    print(StandardScaler().fit_transform(data1.reshape(-1, 1)))
    print(list3)
    print(StandardScaler().fit_transform(list3))
    dataset,average,stddev = ZeroStandardFunc(list3)
    print(ZeroStandardInverseFunc(dataset,average,stddev))

    # 归一化
    print(list3)
    dataset,diff,min = MinMaxScalerFunc(list3)
    print(dataset)
    print(MinMaxScalerInverseFunc(dataset, diff, min))
    print(MinMaxSupervisedFunc(list4, [2]))
    
    # 差分
    print(list3)
    dataset, firstItem = DifferenceFunc(list3)
    print(dataset)
    print(DifferenceInverseFunc(firstItem, dataset))
    
    
    


    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #dataset = scaler.fit_transform(pd.DataFrame(list1))
    #print(dataset)
    #dataset = scaler.inverse_transform(dataset)
    #print(dataset.tolist())
    dataset, firstItem = difference(list2)
    print(dataset)
    dataset = inverse_difference(firstItem, dataset)
    print(dataset)
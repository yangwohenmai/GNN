import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import csv

import sys
sys.path.append('..')

# 获取指定列数据
def GetColsData(path,cols=[0,1]):
    dataframe = read_csv(path, usecols=cols, engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    print(dataset)
    return dataset

def GetAllDataStr(path):
    dataframe = read_csv(path,dtype='str')
    dataset = dataframe.values
    #dataset = dataset.astype('float64')
    print(dataset)
    return dataset

def GetAllDataFloatWithoutHeader(path):
    dataframe = read_csv(path,dtype='float64')
    dataset = dataframe.values
    return dataset

def GetAllDataFloatWithHeader(path):
    dataframe = read_csv(path,dtype='float64', header=0)
    #print(dataframe)
    return dataframe

# 创建csv，如果已有，则追加一列数据
def AddNewLineToCsv1(path, dataDic):
    dataList = list()
    for data in dataDic.keys():
        dataList.append(dataDic[data])
    with open(path,'a+',newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        csv_write.writerow(dataList)

# 创建csv, 如果已有，则追加一列数据
# 文件名filename, 写入dataDic(字典)
def AddNewLineToCsv(filename, dataDic, path=''):
    full_path = '{0}{1}{2}'.format(path, filename, '.csv')
    str = ''
    for data in dataDic.keys():
        str += dataDic[data] + ','
    str = str[:-1]
    file = open(full_path, 'a+')
    file.write('{0}'.format(str))
    file.write('\n')
    file.close()


if __name__ == '__main__':
    #GetAllDataFloatWithoutHeader(r'..\data\BLJJData.csv')
    #GetColsData("")
    AddNewLineToCsv1("file_name.csv", {'total_assets': 197182488.4, 'available_asset': 49592219.45000002, 'shizhi': 94377801.0, 'yingkui': 609527.2199100035})
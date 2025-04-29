from pandas import DataFrame
from pandas import concat
from numpy import array
from numpy import hstack
import numpy
"""
构建监督学习型数据类库
"""
# 将单列list格式数据转换成（输入/输出）监督学习数据
# data：单列数据
# n_in：用于构造输入序列(t-n, ... t-1)，n_in表示每行监督学习数据的长度，如n_in=9，可构造9->1。n_in=0表示停用
# n_out：用于构造输出序列(t, t+1, ... t+n)，n_out表示输出向后跳跃长度，如n_out=3，表示用前n_in天数据预测3天后那一天的数据。n_out=0表示停用
def series_to_supervised(data, n_in, n_out=0):
	df = DataFrame(data)
	cols = list()
	# 得到(t-n, ... t-1, t)：从n_in到-1循环，步长为-1。每次将data向下移动i行，将移动过的数据添加到cols数组中
	for i in range(n_in, -1, -1):
		cols.append(df.shift(i))
	# 得到(t, t+1, ... t+n)：默认n_out=0，不执行该循环
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 将位移过的各个列，横向拼接到一起
	agg = concat(cols, axis=1)
	#print(agg)
	# 删除带有null数据的行
	agg.dropna(inplace=True)
	# 每一行的前n-1列作为输入值，最后一列作为输出值
	return agg.values[:, :-1], agg.values[:, -1]




# 按照比例分割训练集和测试集
def train_test_split(data, percentage):
    n_test = int(len(data) * percentage)
    return data[:-n_test], data[-n_test:]


# 将二维列表 转换成 二维数组
# 二维列表dataset每行数据代表一个特征所包含的所有数值，具有n个特征的dataset则有n行数据
# 若二维列表dataset每行数据表示一个时间步所包含的n个特征值，则要进行转置处理
# transposition:根据需要判断数据是否要转置
def datahstack(dataset, transposition = False):
    # 根据需要，对数组转置处理
    if transposition:
        dataset = array(dataset).T
    newlist = []
    for i in range(len(dataset)):
        dataset[i] = array(dataset[i])
        newlist.append(dataset[i].reshape(len(dataset[i]),1))
    #print(newlist)
    array2D = hstack(newlist)
    #print(array2D)
    return array2D


# (多步+多变量入) -> (单步+单变量出),1D输出
# n_steps:用前n_steps天的数据预测预测后一天的数据
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
def SupervisedData_mm_ss_1D(dataset, n_steps, shift=0):
    X, y = list(), list()
    for i in range(len(dataset)):
        # 计算输入序列的结尾位置
        in_end_ix = i + n_steps_in
        # 计算输出序列的起始位置
        out_start_ix = in_end_ix + shift
        # 如果待预测数据超过（序列长度-1），构造完成
        if out_start_ix > len(dataset) - 1:
            break
        # 取前n_steps行数据的前n-1列作为输入X，第n_steps+1行数据的最后一列作为输出y
        seq_x, seq_y = dataset[i:in_end_ix, :-1], dataset[out_start_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)



# (多步+多变量入) -> (单步+单变量出),2D输出
# n_steps:用前n_steps天的数据预测预测后一天的数据
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
def SupervisedData_mm_ss_2D(dataset, n_steps_in, shift=0):
    X, y = list(), list()
    for i in range(len(dataset)):
        # 计算输入序列的结尾位置
        in_end_ix = i + n_steps_in
        # 计算输出序列的起始位置
        out_start_ix = in_end_ix + shift
        # 如果待预测数据超过（序列长度-1），构造完成
        if out_start_ix > len(dataset) - 1:
            break
        # 取前n_steps行数据的前n-1列作为输入X，第n_steps+1行数据的最后一列作为输出y
        seq_x, seq_y = dataset[i:in_end_ix, :-1], dataset[out_start_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    X = array(X)
    y = array(y)
    # 数据形状重构成多步
    y = y.reshape(y.shape[0], 1)
    return X, y




# (多步+多变量输入)->(多步+单变量输出),输出的字段为最后一列，2D输出
# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
def SupervisedData_mm_ms_2D(sequences, n_steps_in, n_steps_out, shift=0):
    X, y = list(), list()
    for i in range(len(sequences)):
        # 计算输入序列的结尾位置
        in_end_ix = i + n_steps_in
        # 计算输出序列的起始位置
        out_start_ix = in_end_ix + shift
        # 计算输出序列的结尾位置
        out_end_ix = in_end_ix + n_steps_out + shift
        # 判断序列是否结束
        if out_end_ix > len(sequences):
            break
        # 根据算好的位置，取输入输出值
        seq_x, seq_y = sequences[i:in_end_ix, :-1], sequences[out_start_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)



# (多步+多变量输入)->(多步+单变量输出),输出的字段为最后一列，3D输出
# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
def SupervisedData_mm_ms_3D(sequences, n_steps_in, n_steps_out, shift=0):
    X, y = list(), list()
    for i in range(len(sequences)):
        # 计算输入序列的结尾位置
        in_end_ix = i + n_steps_in
        # 计算输出序列的起始位置
        out_start_ix = in_end_ix + shift
        # 计算输出序列的结尾位置
        out_end_ix = in_end_ix + n_steps_out + shift
        # 判断序列是否结束
        if out_end_ix > len(sequences):
            break
        # 根据算好的位置，取输入输出值
        seq_x, seq_y = sequences[i:in_end_ix, :-1], sequences[out_start_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    X = array(X)
    y = array(y)
    # 数据形状重构成多步
    y = y.reshape(y.shape[0], y.shape[1], 1)
    return X, y





# 构造(多步+多变量输入)_(多步+多变量输出),输出包含的所有特征值
# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
def SupervisedData_mm_mm(sequences, n_steps_in, n_steps_out, shift = 0):
    X, y = list(), list()
    for i in range(len(sequences)):
        # 计算输入序列的结尾位置
        in_end_ix = i + n_steps_in
        # 计算输出序列的起始位置
        out_start_ix = in_end_ix + shift
        # 计算输出序列的结尾位置
        out_end_ix = out_start_ix + n_steps_out
        # 判断序列是否结束
        if out_end_ix > len(sequences):
            break
        # 根据算好的位置，取输入输出值
        seq_x, seq_y = sequences[i:in_end_ix, :], sequences[out_start_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
















in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([25,45, 65, 85,105,125,145,165,185])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))


# n_steps_in:输入数据长度
# n_steps_out:输出数据长度
# shift:从距离输入序列结尾位置，上下偏移多少位开始取值
n_steps_in, n_steps_out, shift = 3, 2, 1
#print(dataset)
# (多步+多变量入) -> (单步+单变量出)2D输出
#X,y = SupervisedData_mm_ss_2D(dataset, n_steps_in, 0)
# (多步+多变量输入)_(多步+单变量输出)2D输出
#X, y = SupervisedData_mm_ms_2D(dataset, n_steps_in, n_steps_out, shift)
# (多步+多变量输入)_(多步+单变量输出)3D输出
X, y = SupervisedData_mm_ms_3D(dataset, n_steps_in, n_steps_out, shift)
# (多步+多变量输入)_(多步+单变量输出)
#X, y = SupervisedData_mm_mm(dataset, n_steps_in, n_steps_out, shift)
#print(X)
#print(y)





#in_seq1 = in_seq1.reshape((len(in_seq1), 1))
#in_seq2 = in_seq2.reshape((len(in_seq2), 1))
#out_seq = out_seq.reshape((len(out_seq), 1))
#
#newl = []
#newl.append(in_seq1)
#newl.append(in_seq2)
#newl.append(out_seq)
#
#dataset = hstack(newl)
#print(dataset)
#
#
#X,y = SupervisedData_mm_ss_2D(dataset,4,0)
#print(X)
#print(y)
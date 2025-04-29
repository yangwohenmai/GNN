from matplotlib import pyplot
import matplotlib
"""
画折线类库
"""

"""
画条折线,x轴默认自增排序
linelist:只有一个list数据，根据数据画折线
title：标题
ifgrad：是否显示网格
"""
def DrawLine_List(linelist, title='test', ifgrid=False):
    fig = pyplot.figure()
    ax=fig.add_subplot()
    pyplot.plot(linelist)
    if ifgrid == True:
        pyplot.grid(axis="x")
    ax.legend()
    pyplot.title(title)
    pyplot.show()

"""
画条折线,x轴默认自增排序
linedic:包含多条线段序列的字典
title：标题
ifgrad：是否显示网格
"""
def DrawLineWithout_X(linedic, title='test', ifgrid=False):
    fig = pyplot.figure()
    ax=fig.add_subplot()
    for key in linedic:
        pyplot.plot(linedic[key],label=key)
    if ifgrid == True:
        pyplot.grid(axis="x")
    ax.legend()
    pyplot.title(title)
    pyplot.show()

"""
同一坐标系上使用统一X轴区间
axis_x：使用统一的x轴区间
linedic_axis_y：包含多条待画线段序列的字典
"""
def DrawLineSame_X(axis_x, linedic_axis_y, title='test'):
    pyplot.figure()
    # 在一个图表里显示两根折线
    for key in linedic_axis_y:
        pyplot.plot(axis_x,linedic_axis_y[key])
    print(linedic_axis_y.keys())
    pyplot.legend(linedic_axis_y.keys())
    pyplot.title(title)
    pyplot.show()

"""
同一坐标系上使用不同的X轴区间,画一些不连续的曲线
linedic_axis_x：每个y序列，对应一个自定义x轴区间
linedic_axis_y：包含多条待画线段序列的字典
"""
def DrawLineDiff_X(linedic_axis_x, linedic_axis_y, title='test'):
    pyplot.figure()
    # 在一个图表里显示两根折线
    for key in linedic_axis_y:
        pyplot.plot(linedic_axis_x[key],linedic_axis_y[key])
    pyplot.legend(linedic_axis_y.keys())
    pyplot.title(title)
    pyplot.show()


"""
同一坐标系上使用统一X轴区间，曲线上B/S做标记
axis_x：使用统一的x轴区间
linedic_axis_y：包含多条待画线段序列的字典
buySignalDic：买点"日期-价格"字典
sellSignalDic：卖点"日期-价格"字典
"""
def DrawLine_WithSignal(axis_x, linedic_axis_y, buySignalDic, sellSignalDic, title='test'):
    fig, ax = pyplot.subplots(1)
    # x轴间隔取点显示日期
    pyplot.gca().xaxis.set_major_locator(matplotlib.dates.MonthLocator()) 
    #pyplot.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m")) 
    fig.autofmt_xdate()
    axis_x_new = list()
    for date in axis_x:
        strdate = str(date)
        axis_x_new.append('{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8]))
    # 画n条价格曲线
    for key in linedic_axis_y:
        pyplot.plot(axis_x_new, linedic_axis_y[key])
    # 画买点
    for signalDate in buySignalDic.keys():
        strdate = str(signalDate)
        x = '{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8])
        pyplot.annotate(r'$B$', xy=(x, buySignalDic[signalDate]), xytext=(x, buySignalDic[signalDate]-0.2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #pyplot.annotate(r'B', xy=(x, buySignalDic[signalDate]), xytext=(x, buySignalDic[signalDate]-0.2), arrowprops=dict(facecolor='black', shrink=1, width=0.1))
    # 画卖点
    for signalDate in sellSignalDic.keys():
        strdate = str(signalDate)
        x = '{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8])
        pyplot.annotate(r'$S$', xy=(x, sellSignalDic[signalDate]), xytext=(x, sellSignalDic[signalDate]+0.2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    print(linedic_axis_y.keys())
    pyplot.legend(linedic_axis_y.keys())
    pyplot.title(title)
    pyplot.show()


"""
同一坐标系上使用统一X轴区间，曲线上B/S做标记,显示所有日期
axis_x：使用统一的x轴区间（日期序列）
linedic_axis_y：包含多条待画线段序列的字典
buySignalDic：买点"日期-价格"字典
sellSignalDic：卖点"日期-价格"字典
"""
def DrawLine_WithSignal1(axis_x, linedic_axis_y, buySignalDic, sellSignalDic, title='test'):
    pyplot.figure()
    # x轴间隔取点显示日期
    #pyplot.gca().xaxis.set_major_locator(matplotlib.dates.MonthLocator()) 
    #pyplot.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m")) 
    #fig.autofmt_xdate()
    axis_x_new = list()
    for date in axis_x:
        strdate = str(date)
        axis_x_new.append('{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8]))
    # 画n条价格曲线
    for key in linedic_axis_y:
        pyplot.plot(axis_x_new, linedic_axis_y[key])
    # 画买点
    for signalDate in buySignalDic.keys():
        strdate = str(signalDate)
        x = '{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8])
        pyplot.annotate(r'$B$', xy=(x, buySignalDic[signalDate]), xytext=(x, buySignalDic[signalDate]-0.2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        #pyplot.annotate(r'B', xy=(x, buySignalDic[signalDate]), xytext=(x, buySignalDic[signalDate]-0.2), arrowprops=dict(facecolor='black', shrink=1, width=0.1))
    # 画卖点
    for signalDate in sellSignalDic.keys():
        strdate = str(signalDate)
        x = '{0}-{1}-{2}'.format(strdate[0:4],strdate[4:6],strdate[6:8])
        pyplot.annotate(r'$S$', xy=(x, sellSignalDic[signalDate]), xytext=(x, sellSignalDic[signalDate]+0.2), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    print(linedic_axis_y.keys())
    pyplot.legend(linedic_axis_y.keys())
    pyplot.title(title)
    pyplot.show()


if __name__ == '__main__':
    dic_y = dict()
    dic_x = dict()
    listnum1 = [1,2,3,4,5,6]
    listnum2 = [6,5,4,3,2,1]
    listnum_x1 = [10,20,30,40,50,60]
    listnum_x2 = [-10,-20,-30,-40,-50,-60]
    dic_y['list1'] = listnum1
    dic_y['list2'] = listnum2
    dic_x['list1'] = listnum_x1
    dic_x['list2'] = listnum_x2
    DrawLineDiff_X(dic_x, dic_y)
    DrawLineSame_X(listnum_x1, dic_y)
    DrawLineWithout_X(dic_y)
from matplotlib import pyplot
"""
画折线类库，class类型调用
"""
class DrawLineFunc():
    """
    画条折线,x轴默认自增排序
    linedic:包含多条线段序列的字典
    title：标题
    ifgrad：是否显示网格
    """
    def DrawLineWithout_X(self, linedic, title='test', ifgrid=False):
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
    def DrawLineSame_X(self, axis_x, linedic_axis_y, title='test'):
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
    def DrawLineDiff_X(self, linedic_axis_x, linedic_axis_y, title='test'):
        pyplot.figure()
        # 在一个图表里显示两根折线
        for key in linedic_axis_y:
            pyplot.plot(linedic_axis_x[key],linedic_axis_y[key])
        pyplot.legend(linedic_axis_y.keys())
        pyplot.title(title)
        pyplot.show()





if __name__ == '__main__':
    draw = DrawLineFunc()
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
    draw.DrawLineDiff_X(dic_x, dic_y)
    draw.DrawLineSame_X(listnum_x1, dic_y)
    draw.DrawLineWithout_X(dic_y)
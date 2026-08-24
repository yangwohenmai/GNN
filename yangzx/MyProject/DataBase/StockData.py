import tushare as ts
import datetime as dt
import time
import typing
import sys
import os
import baostock as bs
import math


import matplotlib.pyplot as plt
#import mpl_finance as mpf
import numpy as np
from collections import deque
import pandas as pd

#sys.path.append('..\..')
#print(sys.path)
# 打印文件绝对路径（absolute path）
#print (os.path.abspath(__file__))  
# 打印文件的目录路径（文件的上两层目录）, 这个时候是在 atm 这一层。就是os.path.dirname这个再用了一次
#print (os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))) 
# 要调取其他目录下的文件。 需要在atm这一层才可以
#BASE_DIR=  os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
#print(BASE_DIR)
# 将这个路径添加到环境变量中。
#sys.path.append( BASE_DIR  )



# 取股票均线数据
# maPara: 想要获取的均线窗口值
# period: 想要获取的数据周期
def GetStockMA(stockCode, period=1401, maPara=[10, 20], calDay=100, type="D", benchmark="close"):
    startdate = (dt.datetime.today() - dt.timedelta(period*1)).strftime("%Y%m%d")
    enddate = (dt.datetime.today() - dt.timedelta(period*0)).strftime("%Y%m%d")
    df = ts.pro_bar(ts_code=stockCode, adj='qfq', start_date=startdate, end_date=enddate, ma=maPara)
    
    # 样本点小于40个不计算
    if df is None or df.values is None or len(df.values) < 40:
        print("数据缺失：",stockCode)
        return False
    
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.values)-1,-1,-1):
        # 剔除前n天均线为Nan值的数据
        if i > len(df.values) - maPara[len(maPara)-1]:
            continue
        value = df.values[i]
        OrderDic[value[1]] = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5], \
            'lclose':value[6], 'change':value[7], 'chg':value[8], 'vol':value[9], 'amount':value[10]*1000, \
            'ma_short':value[11], 'ma_v__short':value[12], 'ma_long':value[13], 'ma_v_long':value[14]}
    #print(OrderDic[next(reversed(OrderDic))])
    return OrderDic

# 取股票行情数据
# maPara: 想要获取的均线窗口值
# period: 想要获取的数据周期
def GetStockPriceTushare(stockCode, period=1401, maPara=[10, 20], calDay=100, type="D", benchmark="close"):
    startdate = (dt.datetime.today() - dt.timedelta(period*1)).strftime("%Y%m%d")
    enddate = (dt.datetime.today() - dt.timedelta(period*0)).strftime("%Y%m%d")
    df = ts.pro_bar(ts_code=stockCode, adj='qfq', start_date=startdate, end_date=enddate, ma=maPara)
    
    # 样本点小于40个不计算
    if df is None or df.values is None or len(df.values) < 40:
        print("数据缺失：",stockCode)
        return False
    
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.values)-1,-1,-1):
        # 剔除前n天均线为Nan值的数据
        if i > len(df.values) - maPara[len(maPara)-1]:
            continue
        value = df.values[i]
        OrderDic[value[1]] = {'tdate':value[1], 'open':value[2], 'high':value[3], 'low':value[4], 'close':value[5], \
            'lclose':value[6], 'change':value[7], 'chg':value[8], 'vol':value[9], 'amount':value[10]*1000}
    #print(OrderDic[next(reversed(OrderDic))])
    return OrderDic

# 获取股票(日周月)行情数据
# stockCode: 示例sh.600000
# fields: 返回列字段，分钟数据与日月周数据略有不同，详见文档
# period: 数据周期
# type: 数据类型，默认为d，日k线；d=日k线、w=周、m=月，不区分大小写
# adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
def GetStockPriceDWMBaostock(stockCode, endDate=dt.datetime.today(), period=1401, calDay=100, type="d", benchmark="close"):
    stockCode = stockCode.split('.')[1].lower()+'.'+stockCode.split('.')[0]
    #if startdate == 0:
    #    startdate = (dt.datetime.strptime(endDate,'%Y%m%d') - dt.timedelta(period)).strftime("%Y-%m-%d")
    #    endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    #else:
    #    startdate = dt.datetime.strptime(startdate,'%Y%m%d').strftime("%Y-%m-%d")
    #    endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    if endDate == "":
        endDate = dt.datetime.today()
    else:
        endDate = dt.datetime.strptime(endDate, "%Y%m%d")
    #startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y-%m-%d")
    startDate = (endDate - dt.timedelta(period)).strftime("%Y-%m-%d")
    endDate = endDate.strftime("%Y-%m-%d")
    #"code,date,open,high,low,close,volume,amount,adjustflag"
    df = bs.query_history_k_data_plus(stockCode,"code,date,open,high,low,close,volume,pctChg",start_date=startDate,end_date=endDate,frequency=type, adjustflag="3")
    # 样本点小于40个不计算
    if df is None or df.error_msg != 'success' or len(df.data) == 0:
        print("获取行情数据数据异常：",stockCode)
        return False
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.data)):
        value = df.data[i]
        fields = df.fields
        dic = dict()
        for item in range(len(fields)):
            dic[fields[item]] = value[item]
        OrderDic[value[1]] = dic
    return OrderDic

# 从本地txt缓存读取股票行情数据（返回格式与GetStockPriceDWMBaostock完全一致）
# stockCode: 股票代码，示例000001.SZ（对应缓存文件 000001.SZ.txt）
# endDate: 截止日期，格式"20260801"，空字符串则取文件中最新日期
# period: 向前取多少个自然日
# saveDir: 本地缓存目录，默认 StockDataCache/
def GetStockPriceFromTxt(stockCode, endDate=dt.datetime.today(), period=1401, saveDir=None):
    if saveDir is None:
        saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    filepath = os.path.join(saveDir, f'{stockCode}.txt')
    if not os.path.exists(filepath):
        print(f"本地缓存文件不存在：{filepath}")
        return False
    # 处理endDate参数（与GetStockPriceDWMBaostock保持一致）
    if endDate == "" or isinstance(endDate, dt.datetime):
        if endDate == "":
            endDate = dt.datetime.today()
    else:
        endDate = dt.datetime.strptime(endDate, "%Y%m%d")
    startDate = (endDate - dt.timedelta(period)).strftime("%Y-%m-%d")
    endDateStr = endDate.strftime("%Y-%m-%d")
    # 读取txt文件，按日期范围截取
    OrderDic = typing.OrderedDict()
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # 跳过表头
        fields = header.split(',')
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(',')
            row_date = values[1]  # 第2列是date
            if startDate <= row_date <= endDateStr:
                dic = dict()
                for item in range(len(fields)):
                    dic[fields[item]] = values[item]
                OrderDic[row_date] = dic
    if len(OrderDic) == 0:
        print(f"本地缓存无匹配数据：{stockCode}（{startDate} ~ {endDateStr}）")
        return False
    return OrderDic

# 获取股票(分钟)行情数据
# stockCode: 示例sh.600000
# fields: 返回列字段，分钟数据与日月周数据略有不同，详见文档
# period: 数据周期
# type: 5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写
# adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权
def GetStockPriceMinBaostock(stockCode, startdate, endDate=time.strftime("%Y-%m-%d"), period=1401, calDay=100, type="d", benchmark="close"):
    stockCode = stockCode.split('.')[1].lower()+'.'+stockCode.split('.')[0]
    if startdate == 0:
        startdate = (dt.datetime.today() - dt.timedelta(period)).strftime("%Y-%m-%d")
    else:
        startdate = dt.datetime.strptime(startdate,'%Y%m%d').strftime("%Y-%m-%d")
        endDate = dt.datetime.strptime(endDate,'%Y%m%d').strftime("%Y-%m-%d")
    #"date,time,code,open,high,low,close,volume,amount,adjustflag"
    df = bs.query_history_k_data_plus(stockCode,"date,time,code,open,high,low,close,volume,amount",start_date=startdate,end_date=endDate,frequency=type, adjustflag="2")
    # 样本点小于40个不计算
    if df is None or df.error_msg != 'success' or len(df.data) == 0:
        print("获取行情数据数据异常：",stockCode)
        return False
    # 遍历重构正向时序上的数据
    OrderDic = typing.OrderedDict()
    for i in range(len(df.data)):
        value = df.data[i]
        fields = df.fields
        dic = dict()
        for item in range(len(fields)):
            dic[fields[item]] = value[item]
        OrderDic[value[1]] = dic
    return OrderDic

# 下载指数成分股列表到本地txt缓存
# indexCode: 'hs300'=沪深300，'zz500'=中证500，'sz50'=上证50，'all'=全市场，None=全部下载
# saveDir: 缓存目录，默认 StockDataCache/
def DownloadStockPoolList(indexCode=None, saveDir=None):
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.append(_this_dir)
    import StockPool
    if saveDir is None:
        saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    def _get_all_stock_with_fallback():
        """获取全市场股票列表，非交易日自动往前找最近的交易日"""
        for days_back in range(30):
            query_date = (dt.datetime.today() - dt.timedelta(days=days_back)).strftime('%Y-%m-%d')
            result = StockPool.GetALLStockListBaostock(query_date)
            if len(result) > 0:
                if days_back > 0:
                    print(f'今天非交易日，使用 {query_date} 的全市场数据')
                return result
        return {}
    index_map = {
        'hs300': ('沪深300', StockPool.GetHS300StockListBaostock),
        'zz500': ('中证500', StockPool.GetZZ500StockListBaostock),
        'sz50':  ('上证50',  StockPool.GetSZ50StockListBaostock),
        'all':   ('全市场',  _get_all_stock_with_fallback),
    }
    # 确定要下载的列表
    if indexCode is None:
        download_keys = list(index_map.keys())
    else:
        key = indexCode.lower()
        if key not in index_map:
            print(f'错误：不支持的指数代码 "{indexCode}"，可选: hs300, zz500, sz50, all')
            return
        download_keys = [key]
    lg = bs.login()
    print(f'login respond error_code:{lg.error_code}')
    print(f'login respond error_msg:{lg.error_msg}')
    try:
        for key in download_keys:
            name, func = index_map[key]
            stock_dict = func()
            stock_list = list(stock_dict.keys())
            filepath = os.path.join(saveDir, f'stockpool_{key}.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                for code in stock_list:
                    f.write(code + '\n')
            print(f'{name}成分股列表已保存：{filepath}，共 {len(stock_list)} 只')
    finally:
        bs.logout()

# 从本地txt读取指数成分股列表（返回格式与StockPool.GetHS300StockListBaostock等一致）
# indexCode: 'hs300'=沪深300，'zz500'=中证500，'sz50'=上证50，'all'=全市场
# saveDir: 缓存目录，默认 StockDataCache/
# 返回: dict，如 {'000001.SZ': '000001', '600519.SH': '600519', ...}；文件不存在返回False
def GetStockPoolListFromTxt(indexCode='hs300', saveDir=None):
    if saveDir is None:
        saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    filepath = os.path.join(saveDir, f'stockpool_{indexCode.lower()}.txt')
    if not os.path.exists(filepath):
        print(f"股票池缓存文件不存在：{filepath}")
        return False
    stock_dic = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            code = line.strip()
            if code:
                stock_dic[code] = code.split('.')[0]
    return stock_dic

# 批量下载/增量更新股票行情数据到本地txt缓存
# stock_list: 指定股票代码列表（如['000001.SZ', '600519.SH']），传入则只下载列表中的股票
# indexCode: 按指数下载，'hs300'=沪深300，'zz500'=中证500，'sz50'=上证50，None=全市场
# startYear: 首次下载起始年份（增量更新时自动检测已有数据的最后日期，仅补下载缺失部分）
# saveDir: 本地缓存目录，每只股票一个txt文件（如 000001.SZ.txt）
def DownloadStockData(stock_list=None, indexCode=None, startYear=2020, saveDir=None):
    """
    批量下载/增量更新股票行情数据到本地txt缓存
    三种模式：
      1. 指定stock_list → 只下载列表中的股票
      2. 不传stock_list，指定indexCode → 按指数下载（hs300/zz500/sz50）
      3. 都不传 → 全市场股票下载
    """
    if saveDir is None:
        saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # --- 确定股票列表 ---
    if stock_list:
        # 模式一：指定股票列表
        download_list = list(stock_list)
        print(f'模式一：指定股票列表，共 {len(download_list)} 只')
    else:
        # 模式二/三：需要登录baostock获取码表
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        if _this_dir not in sys.path:
            sys.path.append(_this_dir)
        import StockPool
        lg = bs.login()
        print(f'login respond error_code:{lg.error_code}')
        print(f'login respond error_msg:{lg.error_msg}')
        try:
            if indexCode:
                # 模式二：按指数下载
                index_map = {
                    'hs300': ('沪深300', StockPool.GetHS300StockListBaostock),
                    'zz500': ('中证500', StockPool.GetZZ500StockListBaostock),
                    'sz50':  ('上证50',  StockPool.GetSZ50StockListBaostock),
                }
                key = indexCode.lower()
                if key not in index_map:
                    print(f'错误：不支持的指数代码 "{indexCode}"，可选: hs300, zz500, sz50')
                    return
                name, func = index_map[key]
                download_dict = func()
                download_list = list(download_dict.keys())
                print(f'模式二：按{name}下载，成分股共 {len(download_list)} 只')
            else:
                # 模式三：全市场下载
                today_str = dt.datetime.today().strftime('%Y-%m-%d')
                download_dict = StockPool.GetALLStockListBaostock(today_str)
                download_list = list(download_dict.keys())
                print(f'模式三：全市场下载，共 {len(download_list)} 只')
        finally:
            bs.logout()

    # --- 遍历下载/增量更新 ---
    fields = "code,date,open,high,low,close,volume,pctChg"
    success_count = 0
    skip_count = 0
    fail_count = 0
    total = len(download_list)

    lg = bs.login()
    print(f'login respond error_code:{lg.error_code}')
    print(f'login respond error_msg:{lg.error_msg}')
    try:
        # 先查询一次参考股票，获取baostock当前最新交易日（避免逐只判断时产生无效网络请求）
        # 往前查7天，防止当天未收盘或节假日无数据的情况
        latest_trading_date = dt.datetime.today().strftime('%Y-%m-%d')
        ref_start = (dt.datetime.today() - dt.timedelta(days=7)).strftime('%Y-%m-%d')
        ref_end = dt.datetime.today().strftime('%Y-%m-%d')
        ref_rs = bs.query_history_k_data_plus(
            'sh.000001', 'code,date',
            start_date=ref_start, end_date=ref_end,
            frequency='d', adjustflag='3'
        )
        if ref_rs.error_msg == 'success' and ref_rs.data and len(ref_rs.data) > 0:
            latest_trading_date = ref_rs.data[-1][1]
            print(f'baostock最新交易日: {latest_trading_date}')
        else:
            print(f'查询最新交易日失败，使用今天日期: {latest_trading_date}')

        for i, code in enumerate(download_list):
            # 转换代码格式：000001.SZ → sz.000001
            parts = code.split('.')
            if len(parts) != 2:
                print(f'[{i+1}/{total}] {code} 代码格式错误，跳过')
                fail_count += 1
                continue
            bs_code = parts[1].lower() + '.' + parts[0]
            filepath = os.path.join(saveDir, f'{code}.txt')

            # 检查本地文件，获取最后日期（用于增量判断）
            last_date = None
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        if last_line:
                            last_date = last_line.split(',')[1]  # 第2列是date
                except Exception as e:
                    print(f'[{i+1}/{total}] {code} 读取本地文件异常: {e}，将重新下载')

            # 计算下载范围
            if last_date:
                # 增量更新：本地最后日期 >= baostock最新交易日 → 数据已是最新，跳过
                if last_date >= latest_trading_date:
                    print(f'[{i+1}/{total}] {code} 数据已是最新（{last_date}），跳过')
                    skip_count += 1
                    continue
                # 从最后日期的下一天开始，下载到最新交易日
                start_date = (dt.datetime.strptime(last_date, '%Y-%m-%d') + dt.timedelta(days=1)).strftime('%Y-%m-%d')
                end_date = latest_trading_date
            else:
                # 首次下载：从startYear开始
                start_date = f'{startYear}-01-01'
                end_date = latest_trading_date

            # 查询baostock增量数据
            df = bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3"
            )

            if df.error_msg != 'success':
                print(f'[{i+1}/{total}] {code} 查询失败: {df.error_msg}')
                fail_count += 1
                continue

            new_rows = df.data
            if not new_rows or len(new_rows) == 0:
                print(f'[{i+1}/{total}] {code} 无新增数据（{start_date} ~ {end_date}）')
                skip_count += 1
                continue

            # 写入文件（新建文件写表头，已有文件追加数据）
            is_new_file = not os.path.exists(filepath)
            if not is_new_file:
                # 检查文件末尾是否有换行符，没有则补一个，防止新数据与最后一行拼接
                with open(filepath, 'rb') as f:
                    f.seek(0, 2)  # 移到文件末尾
                    if f.tell() > 0:
                        f.seek(-1, 2)  # 回退一个字节
                        if f.read(1) != b'\n':
                            is_need_newline = True
                        else:
                            is_need_newline = False
                    else:
                        is_need_newline = False
            with open(filepath, 'a', encoding='utf-8') as f:
                if is_new_file:
                    f.write(fields + '\n')
                elif is_need_newline:
                    f.write('\n')
                for row in new_rows:
                    f.write(','.join(row) + '\n')

            success_count += 1
            print(f'[{i+1}/{total}] {code} 新增 {len(new_rows)} 条（{start_date} ~ {end_date}）')

            # 每下载完一只股票休眠1秒，避免请求过于频繁
            time.sleep(1)

    except Exception as e:
        print(f'下载过程异常: {e}')
    finally:
        bs.logout()

    # 汇总
    print(f'\n{"=" * 50}')
    print(f'下载完成！成功更新: {success_count}，跳过(已是最新): {skip_count}，失败: {fail_count}，总计: {total}')
    print(f'缓存目录: {os.path.abspath(saveDir)}')

# 按日期批量更新股票行情数据到本地txt缓存（高效模式，一次获取某天所有股票数据）
# start_date: 开始日期，如 '2026-08-01' 或 '20260801'
# end_date: 结束日期
# saveDir: 本地缓存目录，每只股票一个txt文件（如 000001.SZ.txt）
# 注意：只更新已有txt文件的股票，没有文件的股票会跳过
def UpdateStockDataByDate(start_date, end_date, saveDir=None):
    """
    按日期批量更新股票行情数据（高效模式）
    使用 query_daily_history_k_AStock 一次获取某天所有股票数据
    相比 DownloadStockData（按股票遍历），网络请求次数大幅减少
    
    示例：
      # 更新 2026-08-20 到 2026-08-22 的所有股票数据
      UpdateStockDataByDate('2026-08-20', '2026-08-22')
      
      # 也支持无分隔符格式
      UpdateStockDataByDate('20260820', '20260822')
    """
    if saveDir is None:
        saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'StockDataCache')
    if not os.path.exists(saveDir):
        print(f'缓存目录不存在: {saveDir}')
        return
    
    # 标准化日期格式
    def normalize_date(d):
        if '-' in d:
            return d
        elif len(d) == 8:
            return f'{d[:4]}-{d[4:6]}-{d[6:8]}'
        else:
            return d
    
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    
    print(f'\n{"=" * 50}')
    print(f'按日期批量更新股票行情数据')
    print(f'{"=" * 50}')
    print(f'日期范围: {start_date} ~ {end_date}')
    print(f'缓存目录: {os.path.abspath(saveDir)}')
    
    # 获取日期范围内的所有交易日
    lg = bs.login()
    print(f'login respond error_code:{lg.error_code}')
    print(f'login respond error_msg:{lg.error_msg}')
    
    try:
        # 查询交易日列表
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_msg != 'success':
            print(f'查询交易日失败: {rs.error_msg}')
            return
        
        trading_dates = [row[0] for row in rs.data if row[1] == '1']  # is_trading_day=1
        if not trading_dates:
            print(f'日期范围内无交易日: {start_date} ~ {end_date}')
            return
        
        print(f'交易日数量: {len(trading_dates)}')
        print(f'交易日列表: {trading_dates}')
        
        # 字段映射：query_daily_history_k_AStock 返回的字段索引
        # ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 
        #  'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', ...]
        # 文件字段: code,date,open,high,low,close,volume,pctChg
        FIELD_INDEXES = [1, 0, 2, 3, 4, 5, 7, 12]  # 对应 code,date,open,high,low,close,volume,pctChg
        FILE_FIELDS = "code,date,open,high,low,close,volume,pctChg"
        
        total_update = 0
        total_skip = 0
        total_no_file = 0
        
        # 遍历每个交易日
        for trade_date in trading_dates:
            print(f'\n[{trade_date}] 获取所有股票数据...')
            
            # 一次获取当天所有A股的行情数据
            rs = bs.query_daily_history_k_AStock(date=trade_date)
            if rs.error_msg != 'success':
                print(f'  查询失败: {rs.error_msg}')
                continue
            
            if not rs.data or len(rs.data) == 0:
                print(f'  无数据')
                continue
            
            print(f'  获取 {len(rs.data)} 只股票数据')
            
            # 按股票代码分组
            stock_data_map = {}
            for row in rs.data:
                bs_code = row[1]  # sh.600000
                # 转换代码格式：sh.600000 → 600000.SH
                parts = bs_code.split('.')
                if len(parts) != 2:
                    continue
                stock_code = f'{parts[1].upper()}.{parts[0]}'
                stock_data_map[stock_code] = row
            
            # 统计
            day_update = 0
            day_skip = 0
            day_no_file = 0
            
            # 遍历每只股票
            for stock_code, row_data in stock_data_map.items():
                filepath = os.path.join(saveDir, f'{stock_code}.txt')
                
                # 检查文件是否存在
                if not os.path.exists(filepath):
                    day_no_file += 1
                    continue
                
                # 读取文件最后一行，获取最新日期
                last_date = None
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip()
                        if last_line:
                            last_date = last_line.split(',')[1]  # 第2列是date
                except Exception as e:
                    print(f'  {stock_code} 读取文件异常: {e}')
                    continue
                
                # 判断是否需要追加
                if last_date and last_date >= trade_date:
                    day_skip += 1
                    continue
                
                # 提取需要的字段
                extracted = [row_data[i] for i in FIELD_INDEXES]
                
                # 检查文件末尾是否有换行符
                need_newline = False
                with open(filepath, 'rb') as f:
                    f.seek(0, 2)  # 移到文件末尾
                    if f.tell() > 0:
                        f.seek(-1, 2)  # 回退一个字节
                        if f.read(1) != b'\n':
                            need_newline = True
                
                # 追加数据
                with open(filepath, 'a', encoding='utf-8') as f:
                    if need_newline:
                        f.write('\n')
                    f.write(','.join(extracted) + '\n')
                
                day_update += 1
            
            print(f'  更新: {day_update}, 跳过(已是最新): {day_skip}, 无文件: {day_no_file}')
            total_update += day_update
            total_skip += day_skip
            total_no_file += day_no_file
        
    except Exception as e:
        print(f'更新过程异常: {e}')
    finally:
        bs.logout()
    
    # 汇总
    print(f'\n{"=" * 50}')
    print(f'更新完成！')
    print(f'{"=" * 50}')
    print(f'总更新: {total_update} 条')
    print(f'总跳过(已是最新): {total_skip} 条')
    print(f'总跳过(无文件): {total_no_file} 条')

# 主函数
if __name__ == '__main__':
    import StockPool
    print('begin'+str(dt.datetime.now()))

    #region ========== DownloadStockData 调用示例 ==========
    # 模式一：只下载指定股票
    #DownloadStockData(stock_list=['000001.SZ', '600519.SH'])

    # 模式二：按指数下载（hs300=沪深300, zz500=中证500, sz50=上证50）
    #DownloadStockData(indexCode='zz500')

    # 模式三：全市场下载
    # DownloadStockData()

    # 自定义缓存目录和起始年份
    # DownloadStockData(indexCode='hs300', startYear=2018, saveDir='D:/my_cache')
    #endregion

    #region ========== UpdateStockDataByDate 调用示例（按日期批量更新，高效模式） ==========
    # 更新指定日期范围的所有股票数据（只更新已有txt文件的股票）
    # UpdateStockDataByDate('2026-08-20', '2026-08-22')
    # UpdateStockDataByDate('20260820', '20260822')  # 也支持无分隔符格式
    #endregion

    #region ========== GetStockPriceFromTxt 调用示例 ==========
    # 从本地缓存读取行情数据（返回格式与GetStockPriceDWMBaostock完全一致）
    # result = GetStockPriceFromTxt('000001.SZ', '20260824', 1400)
    # if result != False:
    #     keys = list(result.keys())
    #     print(f'数据条数: {len(result)}')
    #     print(f'日期范围: {keys[0]} ~ {keys[-1]}')
    #     print(f'最新一条: {result[keys[-1]]}')
    #endregion

    #region ========== 股票池列表 调用示例 ==========
    # 下载股票池列表到本地（有网时执行一次即可）
    #DownloadStockPoolList()          # 不传参数=一次性下载全部（沪深300+中证500+上证50+全市场）
    # DownloadStockPoolList('hs300')   # 也可以只下载单个

    # 从本地读取股票池列表（无需联网）
    # pool = GetStockPoolListFromTxt('hs300')
    # if pool != False:
    #     print(f'沪深300成分股: {len(pool)} 只')
    #     print(f'前5只: {list(pool.keys())[:5]}')
    #endregion

    sampleCount = 50
    dataCount = 0
    #### 登陆系统 ####
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond error_msg:'+lg.error_msg)
    #GetALLStockListBaostock()
    #GetAllStockListTushare()
    #GetAllStockListTushareBak()

    stockPoolList = StockPool.GetStockPool('',False,'')
    
    for code in StockPool.GetALLStockListBaostock().keys():
        if len(stockPoolList) == 0 or code not in stockPoolList:
            continue
        try:
            # 获取行情数据
            stockPriceDic = GetStockPriceDWMBaostock(code, "")
            if stockPriceDic == False:
                print(code+"行情获取失败")
                continue
            elif len(stockPriceDic) < sampleCount:
                print(code+"低于最小样本限制")
                continue
            else:
                dataCount += 1
                print(code+'已输出,序号:NO.'+str(dataCount))
            if dataCount == 500:
                time.sleep(60)
        except Exception as ex:
            print("失败代码："+code+"，异常信息："+str(ex))
    print("finish")
    input()

    
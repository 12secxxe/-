#导入库
import pandas
import tushare as ts
import numpy as np
import pandas as pd
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout, Dense, LSTM, SimpleRNN
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#计算平均百分比误差
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 指定使用的字体，这里使用的是中文的黑体
rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
#mean_percentage_error 函数用于计算平均百分比误差（Mean Percentage Error, MPE）。
# 平均百分比误差是一种衡量预测值与实际值之间误差的指标，表示为百分比。该函数实现了对实际值和预测值之间的误差计算，并返回其平均百分比值。
def mean_percentage_error(y_true, y_pred):
    # 避免除以零的情况
    mask = y_true != 0
    return np.mean(((y_true - y_pred) / y_true)[mask]) * 100
#计算皮尔逊相关系数
#correlation_coefficient 函数用于计算两个数组之间的皮尔逊相关系数（Pearson Correlation Coefficient），这是一种衡量两个变量线性相关程度的统计量。
# 皮尔逊相关系数的值介于 -1 和 1 之间，其中 1 表示完全正相关，-1 表示完全负相关，0 表示没有线性相关性。
def correlation_coefficient(y_true, y_pred):
    # 使用Pearson相关系数
    corr, _ = pearsonr(y_true, y_pred)
    return corr
    pass
#使用训练好的模型对输入的最新60个数据点进行预测，并将预测结果逆变换为原始数据尺度，返回归一化后的预测结果和实际预测值。
#predict_oneday 函数用于使用训练好的机器学习模型预测一天的数据。该函数接收三个参数：ele、model 和 sc，分别表示输入数据、机器学习模型和数据缩放器。
def predict_oneday(ele,model,sc):
    x_pre = []
        # 将最后60的数据假如x_pre
    x_pre.append(ele)
    x_pre = np.array(x_pre)
    print(x_pre.shape)
    y_pre = model.predict(x_pre)
    y_norm = y_pre
    y_pre = sc.inverse_transform(y_pre)
    print(y_pre)
    return y_norm, y_pre
#next_day 函数用于计算并返回给定日期的下一天，并将其格式化为 "YYYYMMDD" 的字符串格式。
def next_day(year, month, day):
    current_date = datetime(year, month, day)
    next_date = current_date + timedelta(days=1)
    month = str(next_date.month)
    if str(next_date.month).__len__()<2:
        month = "0"+str(next_date.month)
    day = str(next_date.day)
    if str(next_date.day).__len__()<2:
        day = "0"+str(next_date.day)
    return str(next_date.year) + month + day
#这段代码定义了一个函数 next_time_node，用于计算给定时间点的下一个时间节点，并根据频率（fre）进行相应的增加。
def next_time_node(current_time, fre):
    # 将字符串形式的日期转换为 datetime 对象
    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
    if fre == "1min":
        next_time = current_time + timedelta(minutes=1)
    elif fre == "5min":
        next_time = current_time + timedelta(minutes=5)
    elif fre == "15min":
        next_time = current_time + timedelta(minutes=15)
    elif fre == "30min":
        next_time = current_time + timedelta(minutes=30)
    elif fre == "60min":
        next_time = current_time + timedelta(hours=1)
    elif fre == "1day":
        next_time = current_time + timedelta(days=1)
    else:
        return "Invalid frequency"
    # 将结果转换为所需的字符串格式
    next_time_str = next_time.strftime("%Y-%m-%d %H:%M:%S")
    return next_time_str

def predict_next7(train_set_scaler,xlabel,lookback,model,sc,fre):
    resdic = {}
    lasthavedataday = str(xlabel[-1])
    print(lasthavedataday)
    t = str(lasthavedataday)
    print(type(t))
    lasthavedataday = t
    first_time = next_time_node(lasthavedataday,fre)
    print(first_time)
    first = train_set_scaler[len(train_set_scaler) - lookback:, :]
    print(first.shape)
    for day in range(0, 7):
        if day == 0:
            date = str(first_time)
            print(date)
            time_norm_pre, time_pre = predict_oneday(first, model, sc)
            print(time_pre)
            resdic[date] = time_pre[0]
            temp = first[1:, :]
            temp = np.vstack([temp, time_norm_pre])
            # temp = np.array(temp)
            print(temp.shape)
        else:
            # print(date[4:6])
            date = next_time_node(date, fre)
            print(date)
            time_norm_pre, time_pre = predict_oneday(temp, model,sc)
            # print(day_pre)
            temp = np.vstack([temp[1:, :], time_norm_pre])
            resdic[date] = time_pre[0]
    # 你所选的日期未来7次交易的开盘价收盘价
    for key, value in resdic.items():
        print("{}开盘价是{}，收盘价是{}".format(key, value[0], value[1]))
    # 提取日期、开盘价、收盘价数据
    dates = list(resdic.keys())
    opening_prices = [resdic[date][0] for date in dates]
    closing_prices = [resdic[date][1] for date in dates]

    # 绘制折线图
    plt.plot(dates, opening_prices, label='开盘价', marker='o')
    plt.plot(dates, closing_prices, label='收盘价', marker='o')

    # 添加图例、标题和标签
    plt.legend()
    plt.title('未来股价走势图')
    plt.xlabel('日期')
    plt.ylabel('价格')

    # 旋转日期标签，以避免重叠
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle='--', linewidth=0.7)
    # 显示图形
    plt.tight_layout()
    plt.show()
#通过模型对给定的输入数据进行预测，并打印出未来7天每天的开盘价和收盘价。
def predict_week(train_set_scaler,xlabel,lookback,model,sc):
    resdic = {}
    lasthavedataday = str(xlabel[-1])
    print(lasthavedataday)
    t = str(lasthavedataday)
    print(type(t))
    lasthavedataday = t
    first_day = next_day(int(lasthavedataday[0:4]), int(lasthavedataday[4:-2]), int(lasthavedataday[-2:]))
    print(first_day)
    first = train_set_scaler[len(train_set_scaler) - lookback:, :]
    print(first.shape)
    for day in range(0, 7):
        if day == 0:
            date = str(first_day)
            print(date)
            day_norm_pre, day_pre = predict_oneday(first,model,sc)
            print(day_pre)
            resdic[date] = day_pre[0]
            temp = first[1:, :]
            temp = np.vstack([temp, day_norm_pre])
            # temp = np.array(temp)
            print(temp.shape)
        else:
            # print(date[4:6])
            date = next_day(int(date[0:4]), int(date[4:-2]), int(date[-2:]))
            print(date)
            day_norm_pre, day_pre = predict_oneday(temp,model,sc)
            # print(day_pre)
            temp = np.vstack([temp[1:, :], day_norm_pre])
            resdic[date] = day_pre[0]
     # 你所选的日期未来几天的股价是
    for key, value in resdic.items():
        print("{}开盘价是{}，收盘价是{}".format(key, value[0], value[1]))

    # 提取日期、开盘价、收盘价数据
    dates = list(resdic.keys())
    opening_prices = [resdic[date][0] for date in dates]
    closing_prices = [resdic[date][1] for date in dates]
    # 绘制折线图
    plt.plot(dates, opening_prices, label='开盘价', marker='o')
    plt.plot(dates, closing_prices, label='收盘价', marker='o')
    # 添加图例、标题和标签
    plt.legend()
    plt.grid(axis="y", linestyle='--', linewidth=0.7)
    plt.title('未来一周股价走势图')
    plt.xlabel('日期')
    plt.ylabel('价格')
    # 旋转日期标签，以避免重叠
    plt.xticks(rotation=45)
    # 显示图形
    plt.tight_layout()
    plt.show()
    pass
#将给定的数据集划分为训练集和测试集，并对它们进行归一化处理，同时提取特征和标签。
def split_data_norm(stock, lookback):
    front80 = int(stock.shape[0] * 0.8)
    train_set = stock.iloc[:front80, :].values
    test_set = stock.iloc[front80:, :].values
    sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化到0，1之间
    train_set_scaler = sc.fit_transform(train_set)
    test_set_scaler = sc.transform(test_set)
    print(train_set_scaler.shape)
    print(test_set_scaler.shape)
    # 用前60天的数据来预测61天的数据
    x_train, y_train, x_test, y_test = [], [], [], []  # 定义对象
    for i in range(lookback, len(train_set_scaler)):
        x_train.append(train_set_scaler[i - lookback:i, :])
        y_train.append(train_set_scaler[i, :])
        pass
# 对训练集进行打乱，准备训练集和测试集的特征和标签数据，以便用于模型的训练和评估。
    '''
    在机器学习中，打乱数据集通常有以下几个作用： 防止过拟合（Overfitting） 数据集的顺序可能包含一些
    模式或者偏见，例如，如果你的数据集是按类别排序的，模型可能会学习到这种顺序，而不是从数据的特性中
    学习。 这会导致模型在训练集上表现很好，但在新的、未见过的数据上表现很差。 改善泛化能力 通过打乱
    数据，模型在训练过程中会接触到更多不同类别的数据，这样有助于模型更好地泛化到未见过的数据。
    '''
    np.random.seed(7)
    np.random.shuffle(x_train)
    np.random.seed(7)
    np.random.shuffle(y_train)
    np.random.seed(7)
    # 把训练集从list转换成array格式
    x_train, y_train = np.array(x_train), np.array(y_train)
     # 一个2066行，每行包含一个60行1列的矩阵块的矩阵
    print(x_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], lookback, 2))
    # y_train = np.reshape(-1,1)
    print(x_train.shape)
    print(y_train.shape)
    # 对测试集处理
    for i in range(lookback, len(test_set_scaler)):
        x_test.append(test_set_scaler[i - lookback:i, :])
        y_test.append(test_set_scaler[i, :])
        pass
# 把训练集从list转换成array格式
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(x_test)
    print(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], lookback, 2))
    # y_test = np.reshape(-1,1)
    print(x_test.shape)
    print(y_test.shape)
    return [x_train, y_train, x_test, y_test]

if __name__ == "__main__":
    token = 'd689cb3c1d8c8a618e49ca0bb64f4d6de2f70e28ab5f76a867b31ac7'
    ts.set_token(token)
    pro = ts.pro_api()
    start = '20140511'
    end = '20240518'
    ts_code = '600009.SH'
    # 如果分析日线请值重置为空
    fre = ""
    # 分析港股的才有分钟的数据 fre有1min, 5min,15min,30min,60min
    # SSE上交所 SZSE深交所 BSE北交所
    jysdic = {"SH": "SSE", "SZ": "SZSE"}
    try:
        exchange = jysdic[ts_code[-2:]]
        companydf = pro.stock_company(exchange=exchange)
        info = {}
        for index, row in companydf.iterrows():
            if row["ts_code"] == ts_code:
                cols = companydf.columns
                for col in cols:
                    info[col] = [row[col]]
        infodf = pandas.DataFrame(info)
        infodf.to_excel("./公司基本信息.xlsx", index=False)
    except Exception:
        pass
    # 获取公司高管信息
    gaoguandf = pro.stk_managers(ts_code=ts_code)
    gaoguandf.to_excel("./公司高管信息.xlsx", index=False)
    # 获取管理层持股和薪酬
    cghexcdf = pro.stk_rewards(ts_code=ts_code)
    cghexcdf.to_excel("./公司管理层持股和薪酬情况.xlsx", index=False)
    print("done")
    if fre == "":
        if ts_code[-2:] != "HK":
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        else:
            df = pro.hk_daily(ts_code=ts_code, start_date=start, end_date=end)
    else:
        df = pro.hk_mins(ts_code=ts_code, freq=fre, start_date=start, end_date=end)
    print(df)
    df.to_excel('tushare数据.xlsx')
    filepath = 'D:/智能系统/tushare数据.xlsx'
    data = pd.read_excel(filepath)
    # 将数据按照日期进行排序，确保时间序列递增
    if fre == "":
        data = data.sort_values('trade_date')
    else:
        data = data.sort_values('trade_time')
    # 打印前几条数据
    print(data.head())
    # 打印维度
    print(data.shape)
    # 选取开票价收盘价作为特征
    price = data[['open', 'close']]
    # 打印相关信息
    print(price.info())
    print("****************************************************************************")
    lookback = 30
    print(price.shape)
    # train_set
    x_train, y_train, x_test, y_test = split_data_norm(price, lookback)
    # 对训练集和测试集分别进行归一化处理
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    front80 = int(price.shape[0] * 0.8)
    train_set = price.iloc[:front80, :].values
    test_set = price.iloc[front80:, :].values
    sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化到0，1之间
    train_set_scaler = sc.fit_transform(train_set)
    test_set_scaler = sc.transform(test_set)
    # 搭建网络结构
    model = tf.keras.Sequential([
        LSTM(320, return_sequences=True), # 这里是输入层
        # return_sequences=True: 表示 LSTM 层将返回完整的输出序列，而不仅仅是最后一个时间步的输出。
        # 每个时间步的输出都是一个大小为 80 的向量，而不仅仅是一个标量值。
        Dropout(0.2),  # 20% 的 dropout，这有助于防止过拟合。
        LSTM(200), # 这里是隐藏层
        Dropout(0.2),
        Dense(2)  # 这里是预测层，我们要拿某一天前30的开盘价和收盘价两个来该填的开盘价和收盘价，所以预测层的维度为2
    ])
    # 配置训练方法
    # optimizer=keras.optimizers.Adam(0.001) 表示学习率为0.001，过高的学习率可
    # 能会导致权重更新过快越过最优解，过低的学习率可能会导致收敛非常缓慢，
    # 学习率为 0.001 表示每次权重更新的步长是适度的，不会太大导致不稳定性，也不会太小导致
    # 收敛缓慢。
    '''
    损失函数均连续输出： 股票价格是一个连续的数值，可以取任意实数值。回归任务的目标是预测连续的输出，而不是将样本分类到离散的类别中。在股票预测中，我们关心的是股票价格的具体数值，而不是仅仅预测它会涨还是跌。
    问题的性质： 股票市场的变化是一个连续、动态的过程。尽管在某些情况下可以将股票价格的变化分为“涨”和“跌”，但实际上，我们更关心的是价格的具体变化幅度。这使得股票预测更符合回归问题的性质。
    输出空间无限制： 股票价格的可能取值是无限的，而不是有限的类别。在分类任务中，模型通常需要输出离散的类别标签，而回归任务允许模型输出任意实数值，更适合处理股票价格这样的连续变量。
    评估指标的选择： 对于回归任务，通常使用的评估指标包括均方误差（Mean Squared Error，MSE）、平均绝对误差（Mean Absolute Error，MAE）等，这些指标适用于度量预测值与实际值之间的连续差异。
    在股票预测中，我们更关心价格的准确性和接近度，而不仅仅是它的方向。
    '''
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mean_squared_error", metrics=['mean_squared_error'])
    #  考虑较小的 batch_size 可能导致训练过程更不稳定，把默认的32改成64,
    # 10% 的训练数据作为验证集。这在许多情况下是一个合理的初始值
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.1) # validation_split=0.1从训练集在拿10%做验证集
    model.summary()  # 显示网络结构和参数统计信息

    data = {}
    print(history.history['loss'])
    data["训练集loss"] = history.history['loss']
    data['验证集loss'] = history.history['val_loss']
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(axis="y", linestyle='--', linewidth=0.7)
    plt.title('训练集和验证集Loss损失变化')
    plt.show()

    data = {}
    data["训练集的均方误差"] = history.history['mean_squared_error']
    data['验证集的均方误差'] = history.history['val_mean_squared_error']
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(axis="y", linestyle='--', linewidth=0.7)
    plt.title('训练集和验证集的均方误差变化')
    plt.show()

    # 测试结果 利用训练后的网络对测试集进行预测 检测网络性能 并绘制真实值和预测值之间的对比
    print(front80)
    print(x_test.shape)
    print(x_test[-1])
    print(test_set_scaler[-2])
    predict_stock_price = model.predict(x_test)
    '''
    加速收敛： 在训练神经网络等模型时，梯度下降等优化算法的收敛速度可能会受到输入特征值范围的影响。如果不同特征的值处于不同的范围，某些权重更新可能会更快，导致收敛速度不均匀。通过归一化，可以使不同特征具有相似的尺度，有助于更均匀地更新权重，从而提高模型的训练效率。

    避免数值稳定性问题： 在一些优化算法中，如梯度下降，数值范围较大的特征可能导致数值不稳定性，例如溢出或下溢。通过归一化，可以避免这些数值稳定性问题。

    提高模型泛化能力： 归一化有助于提高模型对未见过的数据的泛化能力。如果模型在训练时见过的数据范围远离测试时的数据范围，可能会导致模型在新数据上的性能下降。归一化有助于缩小这种范围差异，提高模型在新数据上的表现。

    降低特征间的尺度差异： 当特征的尺度相差较大时，某些机器学习算法可能对具有较大尺度的特征更为敏感，从而忽略了尺度较小的特征。归一化可以确保所有特征都具有相似的尺度，使算法更平等地对待所有特征。
    '''
# 对预测的数据还原  从 0-1 反归一化到原始范围
    print(predict_stock_price.shape)
    predict_stock_price = sc.inverse_transform(predict_stock_price)
    print(predict_stock_price)
    predict_openprice = predict_stock_price[:, 0:1]
    print(predict_openprice)
    predict_closeprice = predict_stock_price[:, 1:]
    print(predict_closeprice)
    # 对真实的数据还原 反归一化
    real_stock_price = sc.inverse_transform(test_set_scaler[lookback:])
    real_openprice = real_stock_price[:, 0:1]
    real_closeprice = real_stock_price[:, 1:]

    # 画出真实值和预测值直接的对比曲线
    data = pd.read_excel(filepath)
    # 将数据按照日期进行排序，确保时间序列递增
    data = data.sort_values('trade_date')
    xlabel = list(data["trade_date"])[front80 + lookback:]
    xlabel = np.array(xlabel)
    # 清除之前的图例
    fig, axs = plt.subplots(1, 2)
    cou = 0
    for ax in axs.flat:
        if cou == 0:
            ax.plot(real_openprice, color='red', label='实际的股票开盘价')  # 红线表示真实值
            ax.plot(predict_openprice, color='blue', label='预测的股票开盘价')  # 蓝线表示预测值
            ax.set_title('股票开盘价预测值和真实值比较')
            ax.set_xlabel('时间')
            ax.set_ylabel('股票开盘价')
            labels = xlabel[::100]
            tlabels = []
            for i in labels:
                tlabels.append(str(i))
            labels = tlabels
            print(labels.__len__())
            # labels.append("500")
            print(labels)
            ax.set_xlim(0,500)
            c = 0
            values = []
            for i in labels:
                values.append(c)
                c +=100
            ax.set_xticks(values, labels=labels, rotation=30)
            # 设置 x 轴刻度间隔
            # x_ticks = list(xlabel[::100])  # 每隔 50 个数据点显示一个标签
            # print(x_ticks)
            # plt.xticks(ticks=x_ticks,labels=x_ticks)  # 将二维数组转换为一维数组
            ax.grid(axis="y", linestyle='--', linewidth=0.7)
            ax.legend()
            cou += 1
        else:
            ax.plot(real_closeprice, color='red', label='实际的股票收盘价')  # 红线表示真实值
            ax.plot(predict_closeprice, color='blue', label='预测的股票收盘价')  # 蓝线表示预测值
            ax.set_title('股票收盘价预测值和真实值比较')
            ax.set_xlabel('时间')
            ax.set_ylabel('股票收盘价')
            # print(ax.xticks())
            print(xlabel)
            labels = xlabel[::100]
            tlabels = []
            for i in labels:
                tlabels.append(str(i))
            labels = tlabels
            print(labels.__len__())
            # labels.append("500")
            print(labels)
            ax.set_xlim(0, 500)
            c = 0
            values = []
            for i in labels:
                values.append(c)
                c += 100
            ax.set_xticks(values, labels=labels, rotation=30)
            ax.grid(axis="y", linestyle='--', linewidth=0.7)
            ax.legend()
            pass
    plt.legend()
    plt.show()

    # 通过多个指标来评价预测的结果
    # 均方误差（Mean Squared Error，MSE）：
    mse = mean_squared_error(real_openprice, predict_openprice)
    #  均方根误差（Root Mean Squared Error，RMSE）
    rmse = math.sqrt(mse)
    # 平均绝对误差
    mae = mean_absolute_error(real_openprice, predict_openprice)
    # 平均百分比误差（Mean Percentage Error，MPE）： 衡量模型的预测百分比误差的平均值
    # 相关系数（Correlation Coefficient）： 衡量实际值和预测值之间的线性关系强度和方向。相关系数的取值范围在 [−1,1]，越接近1表示预测与实际的线性关系越强。
    mpe = mean_percentage_error(real_openprice, predict_openprice)
    # corr = correlation_coefficient(real_stock_price.reshape(1,-1)[0], predicted_stock_price)
    r_s = r2_score(real_openprice, predict_openprice)
    print(
        f"开盘价预测均方误差为{mse}, 均方误差的开方为{rmse}， 平均绝对误差为{mae}, 平均百分比误差为{mpe}%，准确率：{r_s}")
    mse = mean_squared_error(real_closeprice, predict_closeprice)
    #  均方根误差（Root Mean Squared Error，RMSE）
    rmse = math.sqrt(mse)
    # 平均绝对误差
    mae = mean_absolute_error(real_closeprice, predict_closeprice)
    # 平均百分比误差（Mean Percentage Error，MPE）： 衡量模型的预测百分比误差的平均值
    mpe = mean_percentage_error(real_closeprice, predict_closeprice)
    r_s = r2_score(real_closeprice, predict_closeprice)
    print(
        f"收盘价价预测均方误差为{mse}, 均方误差的开方为{rmse}， 平均绝对误差为{mae}, 平均百分比误差为{mpe}%，准确率：{r_s}")
    # 实现预测 预测数据后tu.py两天的收盘价
    if fre == "":
        predict_week(train_set_scaler, xlabel, lookback, model, sc)
    else:
        predict_next7(train_set_scaler,xlabel,lookback,model,sc,fre)
import numpy as np #向量与矩阵
import pandas as pd #表格与数据处理
import matplotlib.pylab as plt #绘图
from matplotlib import pyplot
from matplotlib.pylab import rcParams
#rcParams设定好画布的大小
from pandas._config import dates

rcParams['figure.figsize'] = 15, 6
data = pd.read_csv('D:/TSFrame-time-series-forecasting-master/AirPassengers.csv')
print(data.head())
#预览数据前五项
print('\n Data types:')
print(data.dtypes)
#预览数据格式
from datetime import datetime
dateparse = lambda dates_in:datetime.strptime(dates_in,'%Y-%m')
#parse_dates 表明选择数据中的哪个column作为date-time信息，
#index_col 告诉pandas以哪个column作为 index
#date_parser 配合parse_dates使用，是解析日期格式的函数，使一个string转换为一个datetime变量
data = pd.read_csv('D:/TSFrame-time-series-forecasting-master/AirPassengers.csv',parse_dates=['Month'],index_col='Month',date_parser=dateparse)
#将Month的类型变为datatime，同时将其作为索引index
print (data.head())
print (data.index)
#预览数据前五项及索引
#接下来判断时序数据的稳定性
from statsmodels.tsa.stattools import adfuller
#准备进行迪基-福勒检验Augmented Dickey-Fuller(ADF) Test
def test_stationarity(timeseries,):
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    #以一年为一个窗口，每一个时间t的值用涵盖自身的前面十二个月的均值代替，标准差同理,即进行均值与标准差计算
    #绘图
    fig = plt.figure()
    fig.add_subplot()
    orig = plt.plot(timeseries,color = 'blue',label = 'original')
    mean = plt.plot(rolmean,color = 'red',label = 'rolling mean')
    std = plt.plot(rolstd,color = 'black',label = 'Rolling stanard deviation')
    #设置颜色及图像标签
    plt.legend(loc = 'best')#给图像加上图例并设置图例位置
    plt.title('Rolling Mean & Standard Deviation')#设置图像标题
    plt.show(block=False)#利用block=False参数,mataplotlib绘图显示同时继续运行下面的代码
    #Dickey - Fuller(ADF)Test
    print('Results of Dicker-Fuller Test:')
    dftest = adfuller(timeseries,autolag = 'AIC')#增强Dickey-Fuller单位根检验，选择滞后数以最小化相应的信息标准
    #dftest的输出前五项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key] = value
    print (dfoutput)
ts = data['#Passengers']
test_stationarity(ts)
#由图像可见，数据的均值与标准差都具有越来越大的趋势，是不稳定的
#Test Statistic的值如果比Critical Value (5%)小则满足稳定性需求
#p-value越低（理论上需要低于05）证明序列越稳定。
#接下来，使用对数化方法
ts_log = np.log(ts)
#1.使用移动平均方法，作图分析
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log,color='blue',label='ts_log')
plt.plot(moving_avg,color='red',label='moving_avg')
plt.legend(loc='best')
plt.show(block=False)
#作差
ts_log_moving_avg_diff = ts_log-moving_avg
ts_log_moving_avg_diff.head(12)
#从采用过去12个月的值开始，滚动平均法还没有对前11个月的值定义，可以看到前11个月是NaN值，应排除后测试稳定性
ts_log_moving_avg_diff.dropna(inplace = True)#利用dropna函数滤除缺失数据
test_stationarity(ts_log_moving_avg_diff)
#2.使用指数加权移动平均方法
expweighted_avg = pd.Series(ts_log).ewm(halflife=12).mean()#使用12个月的半衰期表示权重
#作差
ts_log_ewma_diff=ts_log-expweighted_avg
test_stationarity(ts_log_ewma_diff)
#3.使用差分化方法检测去除季节性
#构造一个能进行多次差分运算的函数
def data_diff(ts,k):
    i=0
    while i<k:
        ts=ts-ts.shift()
        i=i+1
    return ts
#一阶差分
ts_log_1_diff=data_diff(ts_log,1)
ts_log_1_diff.dropna(inplace=True)
test_stationarity(ts_log_1_diff)
#二阶差分
ts_log_2_diff=data_diff(ts_log,2)
ts_log_2_diff.dropna(inplace=True)
test_stationarity(ts_log_2_diff)
#4.使用分解法检测去除季节性
#分解(decomposing) 可以用来把时序数据中的趋势和周期性数据都分离出来
from statsmodels.tsa.seasonal import seasonal_decompose #调用python的stasmodels库的一种分解方法
def decompose(timeseries): #返回包含三个部分：trend（趋势部分），seasonal（季节性部分）和residual (残留部分)
    decomposition=seasonal_decompose(timeseries)
    trend=decomposition.trend
    seasonal=decomposition.seasonal
    residual=decomposition.resid
    #将图像划为4*1的网格，分区块作图
    plt.subplot(411)
    plt.plot(ts_log,label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(ts_log,label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(ts_log,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(ts_log,label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()#自动调整子图参数，使之填充整个图像区域
    plt.show(block=False)
    return trend,seasonal,residual
trend,seasonal,residual=decompose(ts_log)
residual.dropna(inplace=True)
test_stationarity(residual)
#该方法将original数据拆分成了三份，其中，Trend数据具有明显的趋势性，Seasonality数据具有明显的周期性，Residuals是剩余的部分
#使用ARIMA模型对时序数据进行预测
#1.通过ACF,PACF进行ARIMA（p,d,q）的p,q参数估计（在前面的差分化方法中，一阶差分后数据已经稳定，即d=1
from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(ts_log_1_diff,nlags=20)
lag_pacf=pacf(ts_log_1_diff,nlags=20,method='ols')
#plot ACF(同样创建1*2的网格）
plt.subplot(121)
plt.plot(lag_acf)
#绘制平行于x轴的水平参考线
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_1_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_1_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_1_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show(block=False)
#由图中得知，上下两条灰线之间是置信区间，p的值就是ACF第一次穿过上置信区间时的横轴值。q的值就是PACF第一次穿过上置信区间的横轴值。所以从图中可以得到p=2，q=2
#使用ARLMA模型（ARIMA（2，1，2））
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(ts_log,order=(2,1,2))
results_ARIMA = model.fit(disp=-1)#.disp：True会打印中间过程，直接设置False即可
plt.plot(ts_log_1_diff)
plt.plot(results_ARIMA.fittedvalues,color='red')#fittedvalues返回d次差分后的序列
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_1_diff)**2))
#将模型代入原数据进行预测
#上面的模型的拟合值是对原数据进行稳定化之后的输入数据的拟合，所以需要对拟合值进行相应处理的逆操作，使得它回到与原数据一致的尺度
#ARIMA拟合的其实是一阶差分ts_log_1_diff，predictions_ARIMA_diff[i]是第i个月与i-1个月的ts_log的差值。
#由于差分化有一阶滞后，所以第一个月的数据是空的
predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)#创建序列
print(predictions_ARIMA_diff.head())
#累加现有的diff，得到每个值与第一个月的差分（同log底的情况下）,即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。
predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
#先ts_log_diff => ts_log=>ts_log => ts
#以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第一个月diff数据为空的问题了)
#随后得到了predictions_ARIMA_log => predictions_ARIMA
predictions_ARIMA_log=pd.Series(ts_log.iloc[0],index=ts_log.index)#利用iloc读取第一个值
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)#累加
#缺失值NaN与任何值相加的结果均为NaN，因此用到fill_value，使predictions_ARIMA_log中value的NaN=fill_value，然后与predictions_ARIMA_diff_cumsum中相同索引的value相加
predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.figure()
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
# 加载和预处理数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./data/data.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index('Date', inplace=True)  # 使用Date作为下标

# 设置百分比阈值
percentage_threshold = 0.5
# 检查Word长度并处理异常值
for i in range(1, len(data) - 1):
    current_value = data['Number of  reported results'].iloc[i]
    prev_value = data['Number of  reported results'].iloc[i - 1]
    next_value = data['Number of  reported results'].iloc[i + 1]

    # 计算与前后值的差异
    if (abs(current_value - prev_value) / prev_value > percentage_threshold) or \
       (abs(current_value - next_value) / next_value > percentage_threshold):
        # 将异常值设为NaN
        data['Number of  reported results'].iloc[i] = np.nan
# 检查Word长度并处理异常值
for i in range(len(data)):
    if len(data['Word'].iloc[i]) in [4, 6]:  # 识别异常值
        # 计算前后平均值
        if i > 0 and i < len(data) - 1:  # 确保不越界
            data['Number of  reported results'].iloc[i] = (
                data['Number of  reported results'].iloc[i - 1] +
                data['Number of  reported results'].iloc[i + 1]
            ) / 2
        elif i == 0:  # 如果是第一行
            data['Number of  reported results'].iloc[i] = data['Number of  reported results'].iloc[i + 1]
        elif i == len(data) - 1:  # 如果是最后一行
            data['Number of  reported results'].iloc[i] = data['Number of  reported results'].iloc[i - 1]
# 确保数据没有缺失值
data['Number of  reported results'].ffill(inplace=True)
# 数据平滑，消除每周的季节性影响
data['Smoothed Results'] = data['Number of  reported results'].shift(-7).rolling(window=7).mean()

# 绘制原始数据和处理后的平滑数据
plt.figure(figsize=(12, 6))
plt.plot(data['Number of  reported results'], label='raw data', color='blue', alpha=0.5)
plt.plot(data['Smoothed Results'], label='smoothed data', color='orange')
plt.xlim(pd.to_datetime('2022-01-07'), pd.to_datetime('2022-12-31'))
plt.legend()
plt.title('raw data and smoothed data')
plt.xlabel('date')
plt.ylabel('reported results')
plt.show()

# 进行ADF检验，检查时间序列是否平稳
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Smoothed Results'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])  # 如果p值小于0.05，表示时间序列是平稳的

data['First Differenced'] = data['Smoothed Results'].diff().dropna()
result = adfuller(data['First Differenced'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

data['Second Differenced'] = data['First Differenced'].diff().dropna()
result = adfuller(data['Second Differenced'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# 绘制ACF和PACF图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Smoothed Results'].dropna())  # 用于确定AR模型的阶数q
plot_pacf(data['Smoothed Results'].dropna())  # 用于确定MA模型的阶数p
plt.show()

plot_acf(data['First Differenced'].dropna())  # 用于确定AR模型的阶数q
plot_pacf(data['First Differenced'].dropna())  # 用于确定MA模型的阶数p
plt.show()

plot_acf(data['Second Differenced'].dropna())  # 用于确定AR模型的阶数q
plot_pacf(data['Second Differenced'].dropna())  # 用于确定MA模型的阶数p
plt.show()

# 拟合ARIMA模型
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data['Smoothed Results'].dropna(), order=(1,0,12))
model_fit = model.fit()
print(model_fit.summary())

# 预测未来的结果
forecast = model_fit.forecast(steps=59)  # 预测未来59天
print(forecast)

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(data['Smoothed Results'], label='historical smoothed data', color='orange')
plt.axvline(x=data.index[-1], color='red', linestyle='--', label='predict start')
# 修改预测日期范围，使预测数据从2023年1月开始
forecast_start_date = pd.to_datetime('2023-01-01')  # 设置预测开始日期
plt.plot(pd.date_range(start=forecast_start_date, periods=59, freq='D'), forecast, label='predict results', color='green')  # 预测到2023-3-1
# 修改横轴范围
plt.xlim(pd.to_datetime('2022-01-07'), pd.to_datetime('2023-03-01'))  # 设置横轴范围
plt.title('predict results')
plt.xlabel('date')
plt.ylabel('predicted results')
plt.legend()
plt.show()
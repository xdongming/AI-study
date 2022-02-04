#基于XGBoost的时间序列预测
from xgboost import XGBRegressor as XGBR
from matplotlib import pyplot as plt
import numpy as np
import xlrd

book = xlrd.open_workbook('data.xls')
table = book.sheet_by_index(0)
data = table.col_values(colx=1)
len_data = len(data)
interval = 3
pred_num = 5
data_num = len(data)
w = np.array([1 / 3, 1 / 3, 1 / 3])
xtrain = np.zeros([len_data - interval, interval])
ytrain = []
for i in range(len_data - interval):
    xtrain[i, :] = w * data[i:i + interval]
    ytrain.append(data[i + interval])
reg = XGBR(n_estimators=100).fit(xtrain, ytrain)
pred = []
temp_list = np.zeros([1,pred_num+3])
temp_list[0, 0:3] = data[data_num-3:data_num:1]
for i in range(pred_num):
    inputX = np.zeros([1, interval])
    inputX[0, :] = w * temp_list[0, i:i + interval]
    result = reg.predict(inputX)
    pred.append(result)
    temp_list[0, i+3] = result
x1 = np.linspace(1, len_data, len_data)
x2 = np.linspace(len_data, len_data + pred_num, pred_num + 1)
plt.figure()
plt.plot(x1, data, c='k')
plt.plot(x2, [data[data_num - 1], *pred], c='b')
plt.show()

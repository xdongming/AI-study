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

#xgboost调参模板，n_estimators和eta用网格搜索调好
from time import time
import datetime
dtrain=xgb.DMatrix(Xtrain,Ytrain)
param1 = {'objective':'reg:squarederror',"subsample":1,"max_depth":6,"eta":0.1,"gamma":0,"lambda":1,"alpha":0,"colsample_bytree":1,"colsample_bylevel":1,"colsample_bynode":1}
num_round = 50
time0 = time()
cvresult1 = xgb.cv(param1, dtrain, num_round,nfold=5)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
fig,ax = plt.subplots(1,figsize=(15,8))
#ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1,num_round+1),cvresult1.iloc[:,0],c="red",label="train,original")
ax.plot(range(1,num_round+1),cvresult1.iloc[:,2],c="orange",label="test,original")
param2 = {'objective':'reg:squarederror',"eta":0.1}
param3 = {'objective':'reg:squarederror',"eta":0.1}
time0 = time()
cvresult2 = xgb.cv(param2, dtrain, num_round,nfold=5)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
time0 = time()
cvresult3 = xgb.cv(param3, dtrain, num_round,nfold=5)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
ax.plot(range(1,num_round+1),cvresult2.iloc[:,0],c="green",label="train,last")
ax.plot(range(1,num_round+1),cvresult2.iloc[:,2],c="blue",label="test,last")
ax.plot(range(1,num_round+1),cvresult3.iloc[:,0],c="gray",label="train,this")
ax.plot(range(1,num_round+1),cvresult3.iloc[:,2],c="pink",label="test,this")
ax.legend(fontsize="xx-large")
plt.show()

import pandas as pd
pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)  # 把属性和标签连接成表格

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)  # 划分训练集和测试集

clf.predict(Xtest)  # predict返回每个测试样本的分类/回归结果

cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")   #交叉验证

np.random.random((2,1)).ravel()      #降维
X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]      #升维

'''
数据处理
'''
import pandas as pd

data = pd.read_csv(r'文件所在位置')
data.info()  # 查看数据信息
data.head()  # 查看前n个数据，默认n=5
data.drop(['', ...], inplace=True, axis=1)  # 删除信息，参数为索引，是否覆盖原表
data[''] = data[''].fillna(data[''].mean())  # 用平均值填补缺失，一般用于浮点型数
data = data.dropna(axis=0)  # 删除有缺失值的行

labels = data[''].unique().tolist()  # 查看取值
data[''] = data[''].apply(lambda x: labels.index(x))  # 将标签中的字符串变成索引，即转化为离散数字

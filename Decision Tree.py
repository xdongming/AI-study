from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

dataset = [
    [0, 0, 0, 0, 0, 0, 0.697, 0.460, True],
    [1, 0, 1, 0, 0, 0, 0.774, 0.376, True],
    [1, 0, 0, 0, 0, 0, 0.634, 0.264, True],
    [0, 0, 1, 0, 0, 0, 0.608, 0.318, True],
    [2, 0, 0, 0, 0, 0, 0.556, 0.215, True],
    [0, 1, 0, 0, 1, 1, 0.403, 0.237, True],
    [1, 1, 0, 1, 1, 1, 0.481, 0.149, True],
    [1, 1, 0, 0, 1, 0, 0.437, 0.211, True],
    [1, 1, 1, 1, 1, 0, 0.666, 0.091, False],
    [0, 2, 2, 0, 2, 1, 0.243, 0.267, False],
    [2, 2, 2, 2, 2, 0, 0.245, 0.057, False],
    [2, 0, 0, 2, 2, 1, 0.343, 0.099, False],
    [0, 1, 0, 1, 0, 0, 0.639, 0.161, False],
    [2, 1, 1, 1, 0, 0, 0.657, 0.198, False],
    [1, 1, 0, 0, 1, 1, 0.360, 0.370, False],
    [2, 0, 0, 2, 2, 0, 0.593, 0.042, False],
    [0, 0, 1, 1, 1, 0, 0.719, 0.103, False],
]
attributeList = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
data = []
label = []
for item in dataset:
    data.append(item[0:len(item)-1])
    label.append(item[-1])
    pass
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, label)
print(pd.concat([pd.DataFrame(data), pd.DataFrame(label)], axis=1))
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data, label)
#score = clf.score(Xtest, Ytest)
dot_data = tree.export_graphviz(clf, feature_names=attributeList, class_names=['坏瓜', '好瓜'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph
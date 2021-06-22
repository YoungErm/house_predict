#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz


def createData():
    le = preprocessing.LabelEncoder()
    house_data = pd.read_csv("../processdata/Train.csv")
    data = house_data.iloc[:, 1:-1]
    print("data:\n", data[:5])
    columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Direction', 'Renovation', ]
    for i in columns:
        le.fit(data[i])
        data[i + "Num"] = le.transform(data[i])

    data["PriceNum"] = data["Price"]
    print("data:\n", data[:5])
    data = pd.DataFrame(data)
    data.to_csv("numTrainData.csv")

    ######################################################
    target = {}
    target['Elevator'] = house_data['Elevator']
    le.fit(target['Elevator'])

    target['Elevator' + "Num"] = le.transform(target['Elevator'])
    target = pd.DataFrame(target)
    print("target:\n", target[:5])
    target.to_csv("numTrainTarget.csv")

    ######################################################

    house_data = pd.read_csv("../processdata/test.csv")
    data = house_data.iloc[:, 1:-1]
    print("data:\n", data[:5])
    columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Direction', 'Renovation', ]
    for i in columns:
        le.fit(data[i])
        data[i + "Num"] = le.transform(data[i])
    data["PriceNum"] = data["Price"]
    print("data:\n", data[:5])
    data = pd.DataFrame(data)
    data.to_csv("numTestData.csv")


createData()

data = pd.read_csv("numTrainData.csv")
data = data.iloc[:, 11:].values
target = pd.read_csv("numTrainTarget.csv")
target = target["ElevatorNum"]

# 拆分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
print(Xtrain[:5])
print(Ytrain[:5])

# 建立模型
# entropy是信息增益 id3算法
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30)

clf = clf.fit(Xtrain, Ytrain)

result = clf.predict(Xtrain)
score = clf.score(Xtest, Ytest)
print("score:\n", score)  # 0.9771157167530224
print("result:\n", result)

feature_name = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Direction', 'Renovation', 'Price']
feature_name = [i + "Num" for i in feature_name]

print(clf.feature_importances_)
print([*zip(feature_name, clf.feature_importances_)])

dot_data = tree.export_graphviz(clf,
                                feature_names=feature_name,
                                # class_names=["1", "0"],
                                filled=True,  # 是否填充颜色
                                rounded=True  # 框的形状是否有圆角
                                )

graph = graphviz.Source(dot_data)
graph.view()

testData = pd.read_csv("numTestData.csv")
test = testData.iloc[:, 11:].values
test = pd.DataFrame(test, columns=feature_name)
print(test[:5])
testData['Elevator'] = clf.predict(test)
writeData = testData[['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Direction', 'Renovation', 'Price','Elevator']]
writeData.to_csv("predNonElevator.csv")

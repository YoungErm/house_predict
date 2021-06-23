from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def createData():
    # le = preprocessing.LabelEncoder()
    house_data = pd.read_csv("sumData.csv")
    data = house_data.iloc[:, 1:]
    print("data:\n", data[:5])
    columns = ['Region', 'District', 'Room', 'Hall', 'Direction', 'Renovation', ] #, 'Garden'
    for i in columns:
        print(data[i])
        le = preprocessing.LabelEncoder()
        le.fit(data[i])
        data[i + "Num"] = le.transform(data[i])

    data["ElevatorNum"] = data["Elevator"]
    data["YearNum"] = data["Year"]
    data["SizeNum"] = data["Size"]
    data["FloorNum"] = data["Floor"]
    print("data:\n", data[:5])
    data = pd.DataFrame(data)
    data.to_csv("numTrainData.csv")

    ######################################################
    target = {}
    target['Price'] = house_data['Price']
    target = pd.DataFrame(target)
    print("target:\n", target[:5])
    target.to_csv("numTrainTarget.csv")


createData()

data = pd.read_csv("numTrainData.csv")
data = data.iloc[:, 13:].values
target = pd.read_csv("numTrainTarget.csv")
target = target["Price"]


# 拆分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
print(Xtrain[:5])
print(Ytrain[:5])

clf = DecisionTreeRegressor(random_state=30,max_depth=15)
clf.fit(Xtrain, Ytrain)
score_c = clf.score(Xtest, Ytest)
print("single Tree:{}".format(score_c))


rfc = RandomForestRegressor(random_state=30,oob_score=True,max_depth=15)
rfc.fit(Xtrain, Ytrain)
print("Random Forest oob_scoreTrain:{}".format(rfc.oob_score_))
score_r = rfc.score(Xtest, Ytest)
print("Random Forest oob_scoreTest:{}".format(rfc.oob_score_))
print("Random Forest:{}".format(score_r))


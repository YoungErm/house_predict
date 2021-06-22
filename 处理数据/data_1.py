# 导入第三方模块
import pandas as pd
import numpy as np

# 读入数据
df = pd.read_csv('data.csv')

# 默认是按行删除 即axis=0
df1 = df.dropna()

# print(df1)
df1.to_csv(path_or_buf=r'不含空值数据文档.csv', index=False,
           columns=['Direction', 'District', 'Elevator', 'Floor', 'Garden', 'Id', 'Layout', 'Price', 'Region',
                    'Renovation', 'Size', 'Year'])


# 获取含有空值数据
df2 = df[df.T.isnull().any()]
df2.to_csv(path_or_buf=r'空值数据文档.csv', index=False,
           columns=['Direction', 'District', 'Elevator', 'Floor', 'Garden', 'Id', 'Layout', 'Price', 'Region',
                    'Renovation', 'Size', 'Year'])

 #将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split


x = df1['Direction']#, 'District', 'Floor', 'Garden', 'Id', 'Layout', 'Price', 'Region','Renovation', 'Size', 'Year']

y = df1['Elevator']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    stratify=y,  # 按照标签来分层采样
                                                    shuffle=True,  # 是否先打乱数据的顺序再划分
                                                    random_state=1)  # 控制将样本随机打乱
print(X_train[:5],y_train[:5])

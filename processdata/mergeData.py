#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author ：Mr. Yang
@Date   ：2021/6/22 21:52
@Desc   ：
"""

import pandas as pd
from sklearn import preprocessing

columns = ['Region', 'District', 'Garden','Room','Hall', 'Floor', 'Year', 'Size', 'Direction', 'Renovation',
           'Price', 'Elevator']
dataOrigin = pd.read_csv("不含空值数据文档.csv")
dataOrigin = pd.DataFrame(dataOrigin, columns=columns)
le = preprocessing.LabelEncoder()

le.fit(dataOrigin['Elevator'])

dataOrigin['Elevator'] = le.transform(dataOrigin['Elevator'])

dataFill = pd.read_csv("predNonElevator.csv")
dataFill = pd.DataFrame(dataFill.iloc[:, 1:])

dataFill = pd.DataFrame(dataFill)
print(dataFill)
dataOrigin = pd.DataFrame(dataOrigin)
print(dataOrigin)

sumData = pd.concat([dataFill, dataOrigin])
sumData = pd.DataFrame(sumData)
sumData.to_csv("sumData.csv")
print(sumData)

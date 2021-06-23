#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author ：Mr. Yang
@Date   ：2021/6/22 19:39
@Desc   ：
"""

import pandas as pd


def creatrData(filename, prefix):
    data = pd.read_csv(f"{filename}")
    print(data[:5])

    columns = ['Region', 'District', 'Garden', 'Room', 'Hall', 'Floor', 'Year', 'Size', 'Direction',
               'Renovation',
               'Price', 'Elevator']

    df = pd.DataFrame(data, columns=columns)
    print(df[:5])

    df.to_csv(f"./{prefix}.csv")


creatrData("空值数据文档.csv", "test")
creatrData("不含空值数据文档.csv", "train")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this <training module> is used to generate prediction model, it contains functions:
1) read and transform data
2) generate and save model
3) model evaluation
By default, wo read data from file xlsx, and sheet name is "train"
the 1st column is "content", the 2nd column is "score"

一、此篇为电商评论分数预测模型的训练集部分，主要功能为：
    1) 读取训练集
    2) 生成并保存模型
    3) 模型效果评估
二、训练集默认为xlsx文件，其中sheet名为train。
    sheet中第一列为content列，是文本内容
    sheet中第二列为score列，为分数（需要整数形式）
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
import copy
import xlrd
import datetime
import numpy as np
import os
from sklearn.externals import joblib
import sys
reload(sys)
sys.setdefaultencoding('utf-8')  # @UndefinedVariable


PATH_MAIN = "D:/JavaOuput/Michelin_Ecommerce/PredictModel/"
PATH_DATA = PATH_MAIN+"data/"
PATH_MODEL = PATH_MAIN+"model/"
TRAIN_SHEET = "train"

# read training set
f1 = xlrd.open_workbook(os.path.join(PATH_DATA, "train.xlsx"))
col_score = 1  # column of label
trainfile = f1.sheet_by_name(TRAIN_SHEET)
trainrows = trainfile.nrows
traincols = trainfile.ncols
index = 0

# initialization
train_N_ID = []
train_N = []
train_N_label = []
train_P_ID = []
train_P = []
train_P_label = []

# shuffle positive data to avoid unbalanced training set
for j in range(1, trainrows):
    if trainfile.cell_value(j, col_score) == '5' or trainfile.cell_value(j, col_score) == '4':
        train_P_ID.append(index)
        index += 1
        train_P.append(trainfile.cell_value(j, 0))
        train_P_label.append(trainfile.cell_value(j, col_score))
    else:
        train_N.append(trainfile.cell_value(j, 0))
        train_N_label.append(trainfile.cell_value(j, col_score))

# positive data is at most 5 times of negative data, in order to avoid unbalanced data set
max_length = 5 * len(train_N)
slice_id = random.sample(train_P, min(len(train_P), max_length))
train = copy.copy(train_N)
train_label = copy.copy(train_N_label)
for j in range(len(slice_id)):
    train.append(train_P[slice_id[j]])
    train_label.append(train_P_label[slice_id[j]])

train_label = np.asarray(train_label)

# generate 2-gram model and select features by chi2，then generate final vector space
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), max_df=0.8, binary=True)
ChiVectorizer = SelectKBest(chi2, k=4000)

# apply on training set
fea_train = vectorizer.fit_transform(train)
trainvec = ChiVectorizer.fit_transform(fea_train, train_label)
train_array = np.asarray(trainvec)
train_label_array = np.asarray(train_label)

print("chi2 finished")
print(datetime.datetime.now())

# use multi-algorithm and vote for the final prediction
# logistic regression
lrclf = LogisticRegression(C=1e5)
lrclf.fit(trainvec, train_label)
scores = cross_val_score(lrclf, trainvec, train_label, cv=10).mean()
print("lr train set score:"+str(scores))

# random forest
rfclf = RandomForestClassifier()
rfclf.fit(trainvec, train_label)
scores = cross_val_score(rfclf, trainvec, train_label, cv=10).mean()
print("rf train set score:"+str(scores))

# model storage
joblib.dump(lrclf, os.path.join(PATH_MODEL, "lrclf.m"))
joblib.dump(rfclf, os.path.join(PATH_MODEL, "rfclf.m"))
joblib.dump(ChiVectorizer, os.path.join(PATH_MODEL, "ChiVectorizer.m"))
joblib.dump(vectorizer, os.path.join(PATH_MODEL, "vectorizer.m"))

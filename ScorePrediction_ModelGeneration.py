#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此篇为电商评论分数预测模型的训练集部分，主要功能为：
1、读取训练集
2、生成并保存模型
3、模型效果评估

训练集默认为xlsx文件，其中sheet名为train。
sheet中第一列为content列，是文本内容
sheet中第二列为score列，为分数（需要整数形式）
"""
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
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

# 第一级进行2类的分类，输入第二级进行2类的分类
f1 = xlrd.open_workbook(os.path.join(PATH_DATA, "train.xlsx"))
col_score = 1  # 确定第几列是label，从0开始

# 初始化数据
train_N_ID = []
train_N = []
train_N_label = []
train_P_ID = []
train_P = []
train_P_label = []

# 读入训练集原始文本，并生成标记
trainfile = f1.sheet_by_name(TRAIN_SHEET)
trainrows = trainfile.nrows
traincols = trainfile.ncols
index = 0

# 正面训练集过多，因此随机抽样，再与负面训练集合并为训练集
for j in range(1, trainrows):
    if trainfile.cell_value(j, col_score) == '5' or trainfile.cell_value(j, col_score) == '4':
        train_P_ID.append(index)
        index += 1
        train_P.append(trainfile.cell_value(j, 0))
        train_P_label.append(trainfile.cell_value(j, col_score))
    else:
        train_N.append(trainfile.cell_value(j, 0))
        train_N_label.append(trainfile.cell_value(j, col_score))

dummy_length = 5 * len(train_N)  # 正面训练集数量最多为负面训练集的5倍,避免过度不均衡
slice_id = random.sample(train_P, min(len(train_P), dummy_length))
train = copy.copy(train_N)
train_label = copy.copy(train_N_label)
for j in range(len(slice_id)):
    train.append(train_P[slice_id[j]])
    train_label.append(train_P_label[slice_id[j]])

train_label = np.asarray(train_label)  # .astype('string')

# 通过2-gram生成词袋模型，再通过chi2进行优先选择，确定最后的特征向量空间
# vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), max_df=0.8, binary=True)
# 向量生成器，并导入卡方的选择的特征词作为词袋，生成的向量为0，1的binary形式
# vectorizer = CountVectorizer(analyzer='char_wb',ngram_range=(2,2),max_df=0.8,binary=True,max_features = 5000)
# vectorizer = CountVectorizer(analyzer='char_wb',ngram_range=(2,2),min_df=3,max_df=0.8,binary=True)
# vectorizer = HashingVectorizer(analyzer='char_wb' ,non_negative = True,binary=True)
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), max_df=0.8, binary=True)


ChiVectorizer = SelectKBest(chi2, k=4000)
# 卡方作为特征选择的算法
# ChiVectorizer = SelectPercentile(f_classif, percentile=50)
# ChiVectorizer=SelectKBest(f_classif, k=4000)
# ChiVectorizer=SelectPercentile(chi2, percentile=30)
# ChiVectorizer=SelectPercentile(percentile=40)
# ChiVectorizer=VarianceThreshold()
# ChiVectorizer=GenericUnivariateSelect( mode='percentile')

# 将生成的特征空间方法运用到训练集上
fea_train = vectorizer.fit_transform(train)  # 生成训练集0，1向量
trainvec = ChiVectorizer.fit_transform(fea_train, train_label)  # 由卡方选择的特征生成训练集向量

train_array = np.asarray(trainvec)
train_label_array = np.asarray(train_label)

print("卡方计算完毕！！！")
print(datetime.datetime.now())

# 分别使用逻辑回归和随机森林进行判别，然后采用投票来产出最后结果。并显示出两个模型的score(实际两个模型效果差不多)
lrclf = LogisticRegression(C=1e5)  # lr分类器
lrclf.fit(trainvec, train_label)  # 根据训练集和标签生成模型
scores = cross_val_score(lrclf, trainvec, train_label, cv=10).mean()
print("lr train set score:"+str(scores))

rfclf = RandomForestClassifier()  # 随机森林
rfclf.fit(trainvec, train_label)
scores = cross_val_score(rfclf, trainvec, train_label, cv=10).mean()
print("rf train set score:"+str(scores))


# 模型存储下来，以便在测试集上使用
joblib.dump(lrclf, os.path.join(PATH_MODEL, "lrclf.m"))
joblib.dump(rfclf, os.path.join(PATH_MODEL, "rfclf.m"))
joblib.dump(ChiVectorizer, os.path.join(PATH_MODEL, "ChiVectorizer.m"))
joblib.dump(vectorizer, os.path.join(PATH_MODEL, "vectorizer.m"))

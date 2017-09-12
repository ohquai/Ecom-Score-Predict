#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this <predict module> is used to predict comments' scores, it contains functions:
1) read and transform data
2) predict score and vote for the result
By default, we read data from file xlsx, and sheet name is "test"
it contains only 1 column, is "content"
the result file is format xls, and sheet name is "predict_result"
the 1st column is "content",
the 2nd column is "lr_label" and the 3rd column is "rf_label"
and 4th column is the final result

一、此篇为电商评论分数预测模型的应用部分，主要功能为：
    1) 读取需要预测的数据
    2) 预测分数并存储
二、测试集默认为xlsx文件，其中sheet名为test。
    sheet中只有一列，为content列，是文本内容
三、默认输出结果为xls文件，sheet名为predict_result
    第一列为content，是对应的文本内容
    第二列为lr_label，第三列为rf_label,分别是两种模型预测的分数
    第四列为label，是最终的打分。（具体的投票规则可改）
"""
import xlrd
import xlwt
import datetime
import os
from sklearn.externals import joblib
import sys
reload(sys)
sys.setdefaultencoding('utf-8')  # @UndefinedVariable

PATH_MAIN = "D:/JavaOuput/Michelin_Ecommerce/PredictModel/"
PATH_DATA = PATH_MAIN+"data/"
PATH_MODEL = PATH_MAIN+"model/"
TEST_SHEET = "test"

lrclf = joblib.load(os.path.join(PATH_MODEL, "lrclf.m"))
rfclf = joblib.load(os.path.join(PATH_MODEL, "rfclf.m"))
ChiVectorizer = joblib.load(os.path.join(PATH_MODEL, "ChiVectorizer.m"))
vectorizer = joblib.load(os.path.join(PATH_MODEL, "vectorizer.m"))

# initialization
predict = []
predict_label = []
score_list = []

# read test data
test_file = xlrd.open_workbook(os.path.join(PATH_DATA, "predict.xlsx"))
predictfile = test_file.sheet_by_name(TEST_SHEET)
predictrows = predictfile.nrows
testcols = predictfile.ncols
for j in range(1, predictrows):
    predict.append(predictfile.cell_value(j, 0))

print("data read")
print(datetime.datetime.now())

# predict
fea_predict = vectorizer.transform(predict)
predict_vec = ChiVectorizer.transform(fea_predict)
pred1 = lrclf.predict(predict_vec)
pred2 = rfclf.predict(predict_vec)

print("prediction finished")
print(datetime.datetime.now())

# prepare result file
result_file = xlwt.Workbook(encoding='utf-8')  # 训练集
sheet = result_file.add_sheet('predict_result', cell_overwrite_ok=True)
sheet.write(0, 0, 'content')
sheet.write(0, 1, 'lr label')
sheet.write(0, 2, 'rf label')
sheet.write(0, 3, 'label')

# write result into local file
predict_len = pred1.size
for j in range(0, predict_len):
        sheet.write(j + 1, 0, predict[j])
        sheet.write(j + 1, 1, pred1[j])
        sheet.write(j + 1, 2, pred2[j])
        sheet.write(j + 1, 3, str(min(int(pred1[j]), int(pred2[j]))))
result_file.save(os.path.join(PATH_DATA, "predict.xls"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
此篇为电商评论分数预测模型的应用部分，主要功能为：
1、读取需要预测的数据
2、预测分数并存储

测试集默认为xlsx文件，其中sheet名为test。
sheet中只有一列，为content列，是文本内容

默认输出结果为xls文件，sheet名为predict_result
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

lrclf = joblib.load(os.path.join(PATH_MODEL, "lrclf.m"))  # 保存分类器模型
rfclf = joblib.load(os.path.join(PATH_MODEL, "rfclf.m"))  # 保存分类器模型
ChiVectorizer = joblib.load(os.path.join(PATH_MODEL, "ChiVectorizer.m"))
# 载入卡方选择后的训练集模型，根据该模型来转化测试集为相应的0，1向量
vectorizer = joblib.load(os.path.join(PATH_MODEL, "vectorizer.m"))
# 载入训练集模型，根据该模型来转化测试集为相应的0，1向量

# 初始化数据
predict = []
predict_label = []
score_list = []

# 读入测试集原始文本
test_file = xlrd.open_workbook(os.path.join(PATH_DATA, "predict.xlsx"))
predictfile = test_file.sheet_by_name(TEST_SHEET)
predictrows = predictfile.nrows
testcols = predictfile.ncols
for j in range(1, predictrows):
    predict.append(predictfile.cell_value(j, 0))

print("数据读取完毕")
print(datetime.datetime.now())

# 进行模型判断
fea_predict = vectorizer.transform(predict)  # 生成测试集0,1向量
predict_vec = ChiVectorizer.transform(fea_predict)  # 由卡方选择的特征生成测试集向量
pred1 = lrclf.predict(predict_vec)
pred2 = rfclf.predict(predict_vec)

print("逻辑回归和随机森林计算完毕！！！")
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
        # if int(pred1[j]) < 4 and int(pred2[j]) < 4:
        #     sheet.write(j + 1, 3, str(min(int(pred1[j]), int(pred2[j]))))
        # else:
        #     sheet.write(j + 1, 3, str(max(int(pred1[j]), int(pred2[j]))))
result_file.save(os.path.join(PATH_DATA, "predict.xls"))

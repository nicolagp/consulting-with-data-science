# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:08:49 2019
@author: linos
"""
import pandas as pd
import numpy as np

# substitute path with yours
data = pd.read_csv('train.csv')
# Ideally we would split the data at 34 y/o because that's when causes of death change most dramatically. (according
# to the CDC pdf.
# Because data has been normalized, we split at age = 0.2
young_data = data[data['Ins_Age'] < 0.2]
old_data = data[data['Ins_Age'] >= 0.2]
# Saving csv files to current directory.
# You can change the directory and name of resulting files by altering the value passed to the method.
young_data.to_csv('young_data.csv')
old_data.to_csv('old_data.csv')
data_young = pd.read_csv('young_data.csv')
data_old = pd.read_csv('old_data.csv')
# find continuous variables and change to discrete variables
show = data_young.describe()
data_young.drop(['Family_Hist_5', 'Medical_History_10', 'Medical_History_24'], axis=1, inplace=True)
# Id and product info 2 are irrelevant variables
data_young.drop(['Id', 'Product_Info_2'], axis=1, inplace=True)
# convert continuous variable into discrete variable
data_young['Ins_Age'] = np.where(data_young['Ins_Age'] > data_young['Ins_Age'], 1, 0)
data_young['BMI'] = np.where(data_young['BMI'] > data_young['BMI'].median(), 1, 0)
data_young['Wt'] = np.where(data_young['Wt'] > data_young['Wt'].median(), 1, 0)
data_young['Medical_History_38'] = np.where(
    data_young['Medical_History_38'] > data_young['Medical_History_38'].median(), 1, 0)
data_young['Family_Hist_2'] = np.where(data_young['Family_Hist_2'] > data_young['Family_Hist_2'].median(), 1, 0)
data_young['Family_Hist_3'] = np.where(data_young['Family_Hist_3'] > data_young['Family_Hist_3'].median(), 1, 0)
data_young['Family_Hist_4'] = np.where(data_young['Family_Hist_4'] > data_young['Family_Hist_4'].median(), 1, 0)
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

# perform NB for young dataset
data_test = pd.read_csv('young_data.csv')
x_test = data_test[['Ins_Age', 'Wt', 'BMI', 'InsuredInfo_2', 'Family_Hist_4', 'Medical_History_5', 'Medical_History_20',
                    'Medical_Keyword_3', 'Medical_Keyword_38']].values
y_test = data_test[['Response']].values.ravel()
toomanyNANs = np.isnan(x_test)
x_test[toomanyNANs] = 0
# import complementNB,MultinomialNB
cpl = ComplementNB()
mnb = MultinomialNB()
# train our dataset
cpl.fit(x_test, y_test)
mnb.fit(x_test, y_test)
# perform prediction and find accuracy
y_test_cpl = cpl.predict(x_test)
y_test_mnb = mnb.predict(x_test)
accuracy_testcpl = accuracy_score(y_test, y_test_cpl)
accuracy_testmnb = accuracy_score(y_test, y_test_mnb)
X_young = data_young[
    ['Ins_Age', 'Wt', 'BMI', 'InsuredInfo_2', 'Family_Hist_4', 'Medical_History_5', 'Medical_History_20',
     'Medical_Keyword_3', 'Medical_Keyword_38']].values

Y_young = data_young[['Response']].values.ravel()
NaNs = np.isnan(X_young)
X_young[NaNs] = 0
# train our dataset
cpl.fit(X_young, Y_young)
# perform prediction and find accuracy
y_pred_cpl = cpl.predict(X_young)
accuracy_young_cpl = accuracy_score(Y_young, y_pred_cpl)
# train our dataset
mnb.fit(X_young, Y_young)
# perform prediction and find accuracy
y_pred_young = mnb.predict(X_young)
accuracy_young = accuracy_score(Y_young, y_pred_young)
print(accuracy_young)
# perform NB for old dataset
X_old = data_old[
    ['Ins_Age', 'Ht', 'Wt', 'BMI', 'InsuredInfo_2', 'Family_Hist_4', 'Medical_History_5', 'Medical_History_20',
     'Medical_Keyword_3', 'Medical_Keyword_38']].values
where_are_NaNs = np.isnan(X_old)
X_old[where_are_NaNs] = 0

Y_old = data_old[['Response']].values.ravel()
mnb.fit(X_old, Y_old)
y_pred_old = mnb.predict(X_old)
accuracy_old = accuracy_score(Y_old, y_pred_old)
print(accuracy_old)
# perform NB for all dataset
X_all = data[['Ins_Age', 'Wt', 'BMI', 'InsuredInfo_2', 'Family_Hist_4', 'Medical_History_5', 'Medical_History_20',
              'Medical_History_35', 'Medical_Keyword_3', 'Medical_Keyword_38']].values
are_NaNs = np.isnan(X_all)
X_all[are_NaNs] = 0
Y_all = data['Response']
mnb.fit(X_all, Y_all)
y_pred_all = mnb.predict(X_all)
accuracy_all = accuracy_score(Y_all, y_pred_all)
print(accuracy_all)
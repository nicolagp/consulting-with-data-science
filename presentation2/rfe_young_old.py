import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression

def main():
    data = pd.read_csv('../data/prudential-life-insurance-assessment/train.csv')
    young = data[data['Ins_Age'] < 0.2].copy()
    old = data[data['Ins_Age'] >= 0.2].copy()
    data_rank = get_ranking(data)[:10]
    old_rank = get_ranking(old)[:10]
    young_rank = get_ranking(young)[:10]

def get_ranking(df):
    df = clean(df)
    X = df.drop(['Response', 'Product_Info_2', "Id"], axis=1).values
    X2 = df.drop(['Response', 'Product_Info_2', "Id"], axis=1)
    Y = df['Response']
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    rfe = RFE(lr, n_features_to_select=10)
    rfe.fit(X, Y)

    labels = list(X2.columns)
    ranking = {labels[i]: rfe.ranking_[i] for i in range(len(labels))}
    ranking_list = [(k, ranking[k]) for k in sorted(ranking, key=ranking.get)]
    return ranking_list

def clean(data):
    # dropping less important columns
    data.drop(['Family_Hist_5', 'Medical_History_10', 'Medical_History_24'], axis=1, inplace=True)

    # inputting values based on distribution
    data['Family_Hist_2'].fillna(data['Family_Hist_2'].mean(), inplace=True)
    data['Family_Hist_3'].fillna(data['Family_Hist_3'].mean(), inplace=True)
    data['Family_Hist_4'].fillna(data['Family_Hist_4'].mean(), inplace=True)
    data['Employment_Info_1'].fillna(data['Employment_Info_1'].median(), inplace=True)
    data['Employment_Info_4'].fillna(data['Employment_Info_4'].median(), inplace=True)
    data['Insurance_History_5'].fillna(data['Insurance_History_5'].median(), inplace=True)
    data['Medical_History_1'].fillna(data['Medical_History_1'].median(), inplace=True)
    data['Medical_History_32'].fillna(data['Medical_History_32'].median(), inplace=True)
    data['Employment_Info_6'].fillna(data['Employment_Info_6'].mode()[0], inplace=True)
    data['Medical_History_15'].fillna(data['Medical_History_15'].mode()[0], inplace=True)

    return data

if __name__ == '__main__':
    main()
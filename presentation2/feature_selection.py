import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression

def main():
    data = pd.read_csv('../data/prudential-life-insurance-assessment/train.csv')
    # Cleaning data
    data = clean(data)
    # Get values to run linear regression
    X = data.drop(['Response', 'Product_Info_2'], axis=1).as_matrix()
    Y = data['Response']

    # Linear Regression and RFE
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)

    rfe = RFE(lr, n_features_to_select=10, verbose=3)
    rfe.fit(X, Y)
    print(rfe.ranking_)

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
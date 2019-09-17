from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB, MultinomialNB


def main():
    train = pd.read_csv('../data/prudential-life-insurance-assessment/train.csv')
    train = clean(train)

    # Split into different age groups
    train_original_x, train_original_y, train_young_x, train_young_y, train_old_x, train_old_y = split(train)

    # Get feature selection for each group
    original_features = [i[0] for i in get_ranking(train_original_x, train_original_y)[:10]]
    young_features = [i[0] for i in get_ranking(train_young_x, train_young_y)[:10]]
    old_features = [i[0] for i in get_ranking(train_old_x, train_old_y)[:10]]

    # Run Models
    # Random Forests:
    print("Running Random Forests: ")
    original_rfe = random_forests(train_original_x[original_features], train_original_y)
    young_old_rfe = np.concatenate((random_forests(train_old_x[old_features], train_old_y),
                              random_forests(train_young_x[young_features], train_young_y)))
    print("Accuracy using one model: ", accuracy_score(original_rfe, train_original_y))
    print("Accuracy using two models: ",
    accuracy_score(young_old_rfe, np.concatenate((train_old_y.values, train_young_y.values))))

    print("============================")

    # Naive Bayes
    print("Running Naive Bayes: ")
    original_cpl, original_mnb = naive_bayes(train_original_x[original_features], train_original_y)
    old_cpl, old_mnb = naive_bayes(train_old_x[old_features], train_old_y)
    young_cpl, young_mnb = naive_bayes(train_young_x[young_features], train_young_y)
    young_old_mnb = np.concatenate((old_mnb, young_mnb))
    young_old_cpl = np.concatenate((old_cpl, young_cpl))
    print("Multinomial Naive Bayes Accuracy with 1 model: ", accuracy_score(original_mnb, train_original_y))
    print("Multinomial Naive Bayes Accuracy with 2 models: ",
          accuracy_score(young_old_mnb, np.concatenate((train_old_y.values, train_young_y.values))))
    print("Complement Naive Bayes Accuracy with 1 model: ", accuracy_score(original_cpl, train_original_y))
    print("Complement Naive Bayes Accuracy with 2 models: ",
          accuracy_score(young_old_cpl, np.concatenate((train_old_y.values, train_young_y.values))))

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

def get_ranking(data, response):
    # Drop categorical data
    data.drop('Product_Info_2', axis=1, inplace=True)

    lr = LinearRegression(normalize=True)
    lr.fit(data.values, response)
    rfe = RFE(lr, n_features_to_select=10)
    rfe.fit(data.values, response)

    labels = list(data.columns)
    ranking = {labels[i]: rfe.ranking_[i] for i in range(len(labels))}
    ranking_list = [(k, ranking[k]) for k in sorted(ranking, key=ranking.get)]
    return ranking_list

"""
Returns in this order: original_x, original_y, young_x, young_y, old_x, old_y 
"""
def split(original):
    original_x = original.drop('Response', axis=1)
    original_y = original['Response']
    young_x = original[original['Ins_Age'] < 0.2].drop('Response', axis=1)
    young_y = original[original['Ins_Age'] < 0.2]['Response']
    old_x = original[original['Ins_Age'] >= 0.2].drop('Response', axis=1)
    old_y = original[original['Ins_Age'] >= 0.2]['Response']

    return original_x, original_y, young_x, young_y, old_x, old_y

def random_forests(x, y):
    # fitting model
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    rfc_predict = rfc.predict(x)

    return rfc_predict

"""
Input: x data and y data
Output: CPL predictions, MNB predictions
"""
def naive_bayes(x, y):
    # import complementNB,MultinomialNB
    cpl = ComplementNB()
    mnb = MultinomialNB()
    # train our dataset
    cpl.fit(x, y)
    mnb.fit(x, y)
    # perform prediction and find accuracy
    y_test_cpl = cpl.predict(x)
    y_test_mnb = mnb.predict(x)

    return y_test_cpl, y_test_mnb


if __name__ == '__main__':
    main()
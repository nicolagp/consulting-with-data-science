from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
import pandas as pd
import numpy as np


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
    random_forests(train_original_x[original_features], train_original_y)

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
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=66)
    # fitting model
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)

    # scoring
    rfc_cv_score = cross_val_score(rfc, x, y, cv=10)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

    # # number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # number of features at every split
    # max_features = ['auto', 'sqrt']
    #
    # # max depth
    # max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
    # max_depth.append(None)
    # # create random grid
    # random_grid = {
    #     'n_estimators': n_estimators,
    #     'max_features': max_features,
    #     'max_depth': max_depth
    # }
    # # Random search of parameters
    # rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=5, cv=3, verbose=2,
    #                                 random_state=42, n_jobs=-1)
    # # Fit the model
    # rfc_random.fit(X_train, y_train)
    # # print results
    # print(rfc_random.best_params_)

    # rfc = RandomForestClassifier(n_estimators=rfc_random.best_params_['n_estimators'],
    #                              max_depth=rfc_random.best_params_['max_depth'],
    #                              max_features=rfc_random.best_params_['max_features'])
    rfc = RandomForestClassifier(n_estimators=1400,
                                 max_depth=100,
                                 max_features='auto')

    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    rfc_cv_score = cross_val_score(rfc, x, y, cv=10)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

if __name__ == '__main__':
    main()
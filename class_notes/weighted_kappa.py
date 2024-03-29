import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

def main():
    train = get_train_data()
    responses = get_random_responses(train)
    train['predicted_response'] = responses
    kappa = weighted_kappa(train['predicted_response'], train['Response'], 8)
    print(kappa)

def get_train_data():
    return pd.read_csv('data/prudential-life-insurance-assessment/train.csv')

def get_random_responses(train):
    responses = np.zeros((train.shape[0], 1))
    for i in range(responses.shape[0]):
        responses[i][0] = random.randint(1, 8)
    return responses


"""
predicted: numpy array of predictions for each example
actual: numpy array of correct labels for each example
n: number of rating categories
return: score
"""
def weighted_kappa(predicted, actual, n):
    # calculate observed matrix
    observed = confusion_matrix(actual, predicted)

    # calculate weights matrix
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i][j] = abs(i - j)

    # calculate and normalize expected matrix
    actual_counts = np.array([actual.value_counts()])
    predicted_counts = np.array([predicted.value_counts()])
    expected = np.matmul(np.transpose(actual_counts), predicted_counts)
    normalization = observed.sum() / expected.sum()
    expected = expected * normalization

    # calculate kappa
    kappa = 1 - (np.dot(weights, observed).sum() / np.dot(weights, expected).sum())

    return kappa

if __name__ == '__main__':
    main()
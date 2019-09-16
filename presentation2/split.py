import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # substitute path with yours
    data = pd.read_csv('../data/prudential-life-insurance-assessment/train.csv')

    # Ideally we would split the data at 34 y/o because that's when causes of death change most dramatically. (according
    # to the CDC pdf.
    # Because data has been normalized, we split at age = 0.2
    young_data = data[data['Ins_Age'] < 0.2]
    old_data = data[data['Ins_Age'] >= 0.2]

    # Saving csv files to current directory.
    # You can change the directory and name of resulting files by altering the value passed to the method.
    young_data.to_csv('young_data.csv')
    old_data.to_csv('old_data.csv')


if __name__ == "__main__":
    main()
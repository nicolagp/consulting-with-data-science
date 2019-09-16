import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def main():
    data = pd.read_csv('../data/prudential-life-insurance-assessment/train.csv')
    young = pd.read_csv('young_data.csv')
    old = pd.read_csv('old_data.csv')

    # Create young and old categories
    data['Group'] = ['young' if i else 'old' for i in data['Ins_Age'] < .2]

    # Visualize distributions
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

    a = sns.catplot(x='Group', y='Response', data=data, kind='violin', ax=axes[0])
    b = sns.catplot(hue='Group', x='Response', data=data, kind='count', ax=axes[1])
    plt.close(3)
    plt.close(2)
    plt.show()

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df = pd.read_csv('../data/LA County Crash Data/joint_data.csv')
    df.hist('ACCIDENT_YEAR')

if __name__ == '__main__':
    main()
import pandas as pd


def read_data():
    print(f'Reading CSV file: wdbc.data into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    bc_dataFrame = pd.read_csv("../data/wdbc.data", sep=',', header=None, usecols=list(range(1, 32)))
    print(f"Head of data frame:\n{bc_dataFrame.head()}")
    print(f"First row:\n {bc_dataFrame.loc[0]}")


if __name__ == '__main__':
    read_data()



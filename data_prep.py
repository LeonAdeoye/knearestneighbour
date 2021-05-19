import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl


def read_data():
    print(f'Reading CSV file: wdbc.data into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    bc_dataFrame = pd.read_csv("../data/wdbc.data", sep=',', header=None, usecols=list(range(1, 32)))

    print(f"Head of data frame:\n{bc_dataFrame.head()}")
    print(f"First row:\n {bc_dataFrame.loc[0]}")

    normalized = normalize(bc_dataFrame[1:])
    print(f"First row after normalization:\n {normalized}")

    # group by diagnosis - this returns groupby object.
    diagnosis_groupby = normalized.groupby(by=[1])
    print(f"Count of diagnosis:\n {diagnosis_groupby.size()}")

    normalized[1].hist(bins=2)
    plt.show()



def normalize(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        if column == 1:
            df_scaled[column] = df_scaled[column]
        else:
            df_scaled[column] = (df_scaled[column] - df_scaled[column].min()) / (
                        df_scaled[column].max() - df_scaled[column].min())
    return df_scaled


if __name__ == '__main__':
    read_data()

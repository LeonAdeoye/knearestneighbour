import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def read_data():
    print(f'Reading CSV file: wdbc.data into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    bc_dataFrame = pd.read_csv("../data/wdbc.data", sep=',', header=None, usecols=list(range(1, 32)))
    print(f"Head of data frame:\n{bc_dataFrame.head()}")

    normalized = normalize(bc_dataFrame[1:])
    print(f"First row after normalization:\n {normalized}")

    X = normalized.iloc[:, 1:]
    y = normalized.iloc[:, :1]
    print(f"X:\n {X}")
    print(f"y:\n {y}")

    # group by diagnosis - this returns groupby object.
    diagnosis_groupby = normalized.groupby(by=[1])
    print(f"Count of diagnosis:\n {diagnosis_groupby.size()}")

    normalized[1].hist(bins=2)

    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23122012, stratify=y)

    print(f"After train test split, head of X_train:\n{X_train.head()}")
    print(f"After train test split, head of X_test:\n{X_test.head()}")
    print(f"After train test split, head of y_train:\n{y_train.head()}")
    print(f"After train test split, head of y_test:\n{y_test.head()}")

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    knn = KNeighborsClassifier(n_neighbors=5)


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

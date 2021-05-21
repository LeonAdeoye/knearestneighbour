import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def read_data():
    print(f'Reading CSV file: wdbc.data into dataframe with Panda {pd.__version__}')
    # read a CSV file with a comma as a delimiter, assume no header, and skip the first column
    bc_dataFrame = pd.read_csv("../data/wdbc.data", sep=',', header=None, usecols=list(range(1, 32)))
    print(f"Head after read:\n{bc_dataFrame.head()}")

    # Normalize all numeric data using min-max normalization
    normalized = normalize(bc_dataFrame)
    print(f"Head after normalization:\n {normalized.head()}")

    # Create a correlation table - this is just for code demostration purposes but is not needed below
    print(f"Head of correlation:\n {normalized.corr().head()}")

    # Extract 2-D array of inputs and 1-D array of outputs.
    # Integer-location based indexing for selection by position.
    # All rows and all columns except the first(0)
    X = normalized.iloc[:, 1:]
    # All rows and only the first column
    y = normalized.iloc[:, :1]
    print(f"X:\n {X.head()}")
    print(f"y:\n {y.head()}")

    # group by diagnosis - this returns groupby object.
    diagnosis_groupby = normalized.groupby(by=[1])
    # Use size to get the count.
    print(f"Count of diagnosis:\n {diagnosis_groupby.size()}")

    # split the dataframe into train and test datasets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.175, random_state=23122012, stratify=y)

    print(f"After train test split, head of X_train:\n{X_train.head()}")
    print(f"After train test split, head of X_test:\n{X_test.head()}")
    print(f"After train test split, head of y_train:\n{y_train.head()}")
    print(f"After train test split, head of y_test:\n{y_test.head()}")

    # Get the dimensions of the train and test datsets.
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Configure the KNN model using a k value of 21
    knn_model = KNeighborsClassifier(n_neighbors=15)
    # Fit the training data to the model
    knn_model.fit(X_train, y_train.values.ravel())
    # Using the test data, use the model to make the predictions
    y_prediction = knn_model.predict(X_test)
    # Display arrays of the predictions and test results
    print(f"Prediction: {y_prediction}")
    print(f"Convert dataframe of one column to array of items: {y_test.values.ravel()}")
    # Calculate the accuracy of the data by comparing the known
    # Accuracy of 96% when comparing the results of known test data with the predicted results of the test data
    print("Accuracy: ", metrics.accuracy_score(y_test, y_prediction))

    # Display a cross tabulation of the data
    crosstab_result = pd.crosstab(y_test.values.ravel(), y_prediction, margins=True)
    print(f"Cross tabulation of y_predicted versus y_test:\n{crosstab_result}")

    # Display a histogram of the diagnosis.
    normalized[1].hist(bins=2)
    plt.show()

    # Trying different values of k and calculating mean error into an array
    error = []
    for k in range(1, 40):
        # Configure the KNN model using a k value of 1 to 40
        knn_model = KNeighborsClassifier(n_neighbors=k)
        # Fit the training data to the model
        knn_model.fit(X_train, y_train.values.ravel())
        # Using the test data, use the model to make the predictions
        y_prediction = knn_model.predict(X_test)
        error.append(np.mean(y_prediction != y_test.values.ravel()))

    # Chart shows that the lowest means error are when k = 13 to 15
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error rate K value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
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

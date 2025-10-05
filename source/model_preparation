import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_X_y(df):
    """
    Feature engineering and create X and y
    :param df: pandas dataframe
    :return: (X, y) output feature matrix (dataframe), target (series)
    """
    # Todo: Split data into X and y (using sklearn train_test_split). Return two dataframes
    if 'target' in df.columns:
        X = df.drop('target', axis = 1)
        y = df['target']
    return X, y

def split(X, y, train_size, random_state):
    """Wrapper around sklearn.train_test_split using the provided parameters.

    Returns trainX, testX, trainY, testY with indices reset.
    """
    trainX, testX, trainY, testY = train_test_split(X, y, train_size=train_size, random_state=random_state)
    trainX = trainX.reset_index(drop=True)
    testX = testX.reset_index(drop=True)
    trainY = trainY.reset_index(drop=True)
    testY = testY.reset_index(drop=True)
    return trainX, testX, trainY, testY
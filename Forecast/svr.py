import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from create_dataset import create_dataset
from sklearn.svm import SVR
import tensorflow as tf

def svr_model(new_data, i, look_back, data_partition, cap, target):
    # Filter and clean data
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1[target].values.reshape(-1, 1)  # reshape to ensure 2D

    # Split into train/test
    train_size = int(len(s) * data_partition)
    train, test = s[:train_size], s[train_size:]

    # Create supervised datasets
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Standardize input features and targets
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(trainX)
    y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()  # Ensure 2D then flatten
    X_test = sc_X.transform(testX)
    y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()        # Ensure 2D then flatten

    # Set seeds for reproducibility
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Train SVR model
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)

    # Make predictions
    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)

    # Inverse transform predictions and true values
    y_pred_train_inv = sc_y.inverse_transform(y_pred_train.reshape(-1, 1))
    y_pred_test_inv = sc_y.inverse_transform(y_pred_test.reshape(-1, 1))
    y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    mape = np.mean(np.abs(y_test_inv - y_pred_test_inv) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_test_inv)

    # Print results
    print('MAPE:', mape)
    print('RMSE:', rmse)
    print('MAE:', mae)

    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }

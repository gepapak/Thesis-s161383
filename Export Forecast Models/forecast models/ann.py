import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from create_dataset import create_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def ann_model(new_data, i, look_back, data_partition, cap, target, return_model=False):
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    data_series = data1[target].values.reshape(-1, 1)

    # Split and preprocess
    train_size = int(len(data_series) * data_partition)
    train, test = data_series[:train_size], data_series[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(trainX)
    y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()
    X_test = sc_X.transform(testX)
    y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

    # Define ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

    # Prediction
    y_pred = model.predict(X_test).ravel()
    y_pred_inv = sc_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mape = np.mean(np.abs(y_test_inv - y_pred_inv) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    if return_model:
        return mape, rmse, mae, model
    else:
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

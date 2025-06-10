from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from math import sqrt
import numpy as np
import tensorflow as tf
import os
from create_dataset import create_dataset

def lstm_model(new_data, i, look_back, data_partition, cap, target, return_model=False):
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    s = data1[target].values.reshape(-1, 1)

    train_size = int(len(s) * data_partition)
    train, test = s[:train_size], s[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(trainX)
    y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()
    X_test = sc_X.transform(testX)
    y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Reproducibility
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = Sequential()
    model.add(tf.keras.Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(LSTM(units=128))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(X_train_lstm, y_train, epochs=100, batch_size=64, verbose=0)

    y_pred_test = model.predict(X_test_lstm, verbose=0).ravel()
    y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_test_inv = sc_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

    mape = np.mean(np.abs(y_test_inv - y_pred_test_inv) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_test_inv)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    if return_model:
        return model
    else:
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

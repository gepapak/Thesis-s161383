import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from create_dataset import create_dataset
from PyEMD import CEEMDAN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def ceemdan_lstm_model(new_data, i, look_back, data_partition, cap, target, return_model=False):
    # Filter and prepare data
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    raw_signal = data1[target].values.reshape(-1, 1)

    # CEEMDAN decomposition
    ceemdan = CEEMDAN(epsilon=0.05)
    ceemdan.noise_seed(12345)
    IMFs = ceemdan(raw_signal.flatten())
    imf_df = pd.DataFrame(IMFs).T  # Each IMF is a column

    # LSTM parameters
    epochs = 100
    batch_size = 64
    neurons = 128
    learning_rate = 0.001

    pred_test_list = []
    true_test_list = []

    last_model = None  # to return if needed

    for col in imf_df.columns:
        component = imf_df[col].values.reshape(-1, 1)
        train_size = int(len(component) * data_partition)
        train, test = component[:train_size], component[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(trainX)
        y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()
        X_test = sc_X.transform(testX)
        y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

        # Reshape for LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], look_back, 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], look_back, 1))

        # Seed for reproducibility
        np.random.seed(1234)
        tf.random.set_seed(1234)

        # Build and train LSTM model
        model = Sequential()
        model.add(LSTM(units=neurons, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Save the last trained model (on last IMF)
        last_model = model

        # Predict
        y_pred_test = model.predict(X_test_lstm, verbose=0).ravel()
        y_pred_test_inv = sc_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
        y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        pred_test_list.append(y_pred_test_inv)
        true_test_list.append(y_test_inv)

    # Reconstruct final prediction and true test signal
    y_pred_final = np.sum(pred_test_list, axis=0)
    y_true_final = np.sum(true_test_list, axis=0)

    # Evaluate
    mape = np.mean(np.abs(y_true_final - y_pred_final) / cap) * 100
    rmse = sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae = mean_absolute_error(y_true_final, y_pred_final)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    if return_model:
        return mape, rmse, mae, last_model
    else:
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

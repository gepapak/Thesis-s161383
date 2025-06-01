# ANN Model Function
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from math import sqrt
from create_dataset import create_dataset

def ann_model(new_data, i, look_back, data_partition, cap, target):
    
    
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    datas = data1[target]
    datasets = datas.values.reshape(-1, 1)

    train_size = int(len(datasets) * data_partition)
    train, test = datasets[:train_size], datasets[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(trainX)
    y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()
    X_test = sc_X.transform(testX)
    y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

    # Ensure data is float32 for TensorFlow
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # ✅ Keep 2D shape (no extra dimension)
    trainX1 = X_train
    testX1 = X_test

    # Set random seed for reproducibility
    np.random.seed(1234)
    tf.random.set_seed(1234)
    os.environ['PYTHONHASHSEED'] = '0'
    tf.config.experimental.enable_op_determinism()

    # ✅ Correct Input Shape for ANN
    neuron = 128
    model = Sequential()
    model.add(Dense(units=neuron, activation='relu', input_shape=(trainX1.shape[1],)))  # Fixed input shape
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
    model.fit(trainX1, y_train, verbose=0)

    # Make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1).ravel()

    # Inverse transform predictions
    y_pred_test = sc_y.inverse_transform(y_pred_test.reshape(-1, 1))
    y_test = sc_y.inverse_transform(y_test.reshape(-1, 1))

    # Summarize the fit of the model
    mape = np.mean((np.abs(y_test - y_pred_test)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    print('MAPE:', mape)
    print('RMSE:', rmse)
    print('MAE:', mae)
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }
def karijadi_model(new_data, i, look_back, data_partition, cap, target):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from math import sqrt
    import numpy as np
    import pandas as pd
    import os
    import tensorflow as tf
    import ewtpy
    from PyEMD import CEEMDAN
    import gc
    from tensorflow.keras import backend as K

    # Step 1: Extract the relevant data and apply global scaling
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    original_series = data1[target].values.reshape(-1, 1)

    # Global scaling on full original signal
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(original_series)

    # Step 2: CEEMDAN decomposition
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(scaled_series.flatten())
    ceemdan1 = pd.DataFrame(IMFs).T
    imf1 = ceemdan1.iloc[:, 0]

    # Step 3: EWT on IMF1
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    if df_ewt.shape[1] > 2:
        df_ewt.drop(columns=[2], inplace=True)
    denoised = df_ewt.sum(axis=1)

    # Step 4: Rebuild component matrix (denoised + other IMFs)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    print("new_ceemdan shape:", new_ceemdan.shape)

    pred_test, pred_train = [], []

    # Step 5: Train LSTM for each IMF component
    for col in new_ceemdan:
        dataset = new_ceemdan[[col]].values

        train_size = int(len(dataset) * data_partition)
        train, test = dataset[:train_size], dataset[train_size:]

        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        trainX = trainX.reshape(trainX.shape[0], look_back, 1)
        testX = testX.reshape(testX.shape[0], look_back, 1)

        # Model
        model = Sequential()
        model.add(LSTM(128, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)

        y_pred_train = model.predict(trainX).ravel()
        y_pred_test = model.predict(testX).ravel()

        pred_train.append(y_pred_train)
        pred_test.append(y_pred_test)

        # Memory cleanup
        K.clear_session()
        del model
        gc.collect()

    # Step 6: Sum across IMF predictions
    pred_test = np.array(pred_test).reshape(len(pred_test), -1)
    pred_train = np.array(pred_train).reshape(len(pred_train), -1)
    summed_pred_test = pred_test.sum(axis=0)
    summed_pred_train = pred_train.sum(axis=0)

    # Step 7: Inverse transform the final summed predictions
    summed_pred_test_inv = scaler.inverse_transform(summed_pred_test.reshape(-1, 1))
    summed_pred_train_inv = scaler.inverse_transform(summed_pred_train.reshape(-1, 1))

    # Reconstruct actual y_test from original scaled series
    _, actual_y_test = create_dataset(scaled_series[int(len(scaled_series) * data_partition):], look_back)
    actual_y_test_inv = scaler.inverse_transform(actual_y_test.reshape(-1, 1))

    # Metrics
    mape = np.mean(np.abs(actual_y_test_inv - summed_pred_test_inv) / cap) * 100
    rmse = sqrt(mean_squared_error(actual_y_test_inv, summed_pred_test_inv))
    mae = mean_absolute_error(actual_y_test_inv, summed_pred_test_inv)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }
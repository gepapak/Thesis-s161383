def quantile_forecasting_model(new_data, i, look_back, data_partition, cap, quantiles=[0.1, 0.5, 0.9]):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import ewtpy
    from PyEMD import CEEMDAN
    import gc
    from create_dataset import create_dataset

    def pinball_loss(q):
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
        return loss

    # Step 1: Filter & Scale
    data1 = new_data.loc[new_data['Month'].isin(i)].reset_index(drop=True).dropna()
    original_series = data1['hydro'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(original_series)

    # Step 2: CEEMDAN
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(scaled_series.flatten())
    ceemdan1 = pd.DataFrame(IMFs).T
    imf1 = ceemdan1.iloc[:, 0]

    # Step 3: EWT
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    df_ewt = pd.DataFrame(ewt)
    if df_ewt.shape[1] > 2:
        df_ewt.drop(columns=[2], inplace=True)
    denoised = df_ewt.sum(axis=1)

    # Step 4: New CEEMDAN signal
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    quantile_preds = {q: [] for q in quantiles}

    # Step 5: Train model per IMF + quantile
    for col in new_ceemdan:
        dataset = new_ceemdan[[col]].values
        train_size = int(len(dataset) * data_partition)
        train, test = dataset[:train_size], dataset[train_size:]
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        trainX = trainX.reshape(trainX.shape[0], look_back, 1)
        testX = testX.reshape(testX.shape[0], look_back, 1)

        for q in quantiles:
            tf.keras.backend.clear_session()
            inputs = Input(shape=(look_back, 1))
            x = LSTM(128, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(64)(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            model = Model(inputs, outputs)
            model.compile(loss=pinball_loss(q), optimizer=tf.keras.optimizers.Adam(0.0005))
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(trainX, trainY, epochs=150, batch_size=32, verbose=0, callbacks=[early_stop])
            pred_q = model.predict(testX).ravel()
            quantile_preds[q].append(pred_q)
            del model
            gc.collect()

    final_preds = {}
    for q in quantiles:
        summed_q = np.sum(np.array(quantile_preds[q]), axis=0)
        final_preds[q] = scaler.inverse_transform(summed_q.reshape(-1, 1)).ravel()

    _, actual_y_test = create_dataset(scaled_series[int(len(scaled_series) * data_partition):], look_back)
    actual_y_test_inv = scaler.inverse_transform(actual_y_test.reshape(-1, 1)).ravel()

    # Step 6: Evaluation
    results = {}
    for q in quantiles:
        pred = final_preds[q]
        mae = mean_absolute_error(actual_y_test_inv, pred)
        pinball = np.mean(np.maximum(q * (actual_y_test_inv - pred), (q - 1) * (actual_y_test_inv - pred)))
        results[q] = {
            'MAE': mae,
            'Pinball Loss': pinball
        }

    # Optional Interval Evaluation (P90 - P10)
    if 0.1 in final_preds and 0.9 in final_preds:
        lower = final_preds[0.1]
        upper = final_preds[0.9]
        coverage = np.mean((actual_y_test_inv >= lower) & (actual_y_test_inv <= upper))
        interval_width = np.mean(upper - lower)
        alpha = 0.8
        cwc = interval_width * (1 + 1 * np.exp(-10 * (coverage - alpha)))  # CWC: Coverage Width-based Criterion
        results['interval'] = {
            'Coverage': coverage,
            'Interval Width': interval_width,
            'CWC': cwc
        }

    return final_preds, actual_y_test_inv, results

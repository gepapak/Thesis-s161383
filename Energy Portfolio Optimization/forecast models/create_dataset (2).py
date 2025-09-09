import numpy as np

def create_dataset(dataset, look_back=1, multivariate=False):
    """
    Turns a series (N,1) or matrix (N,F) into (X, y) with right-aligned windows.
    Univariate target is always the first column.
    """
    dataX, dataY = [], []

    if multivariate:
        # X: [i : i+look_back, :], y: dataset[i+look_back, 0]
        for i in range(len(dataset) - look_back):
            X_slice = dataset[i:(i + look_back), :]
            y_value = dataset[i + look_back, 0]
            dataX.append(X_slice)
            dataY.append(y_value)
    else:
        # X: [i : i+look_back, 0], y: dataset[i+look_back, 0]
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)

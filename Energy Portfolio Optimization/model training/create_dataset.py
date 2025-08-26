import numpy as np

# Convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1, multivariate=False):
    dataX, dataY = [], []
    if multivariate:
        for i in range(len(dataset) - look_back):
            X_slice = dataset[i:(i + look_back), :]
            y_value = dataset[i + look_back, 0]  # target is first column
            dataX.append(X_slice)
            dataY.append(y_value)
    else:
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)

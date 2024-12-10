import numpy as np
from scipy.spatial.distance import cdist


# Moving Average
def moving_average(x, n=100):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Normalize List
def normalize_nested_lists(arr):
    normalized_arr = []
    for sublist in arr:
        sublist  = sublist.astype('float64')  # explicitly cast to float64
        row_sums = np.sum(sublist, axis=1, keepdims=True)
        row_sums = np.where(row_sums==0, 1, row_sums)
        np.divide(sublist, row_sums, out=sublist)
        normalized_arr.append(sublist.tolist())
    return normalized_arr

def distance_dots(data, threshold=0.5):
    distances = np.zeros((data.shape[0], 3, 3))
    for i in range(data.shape[0]):
        distances[i] = cdist(data[i], data[i], metric='euclidean')

    temp = np.where(distances == 0, 1e+10, distances)  # replace zero distances with big values
    temp = np.where(temp > threshold, 0, 1)  # threshold the distances
    distances = np.concatenate([distances, temp], axis=2)  # add three more values
    return distances


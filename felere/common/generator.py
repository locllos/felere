import numpy as np

def splitter(X: np.ndarray, y: np.ndarray, n_parts):
  n_samples = X.shape[0]
  split_size = n_samples // n_parts + 1

  for i in range(0, n_samples, split_size):
    yield X[i: min(i + split_size, n_samples)], y[i: min(i + split_size, n_samples)]


def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int):
  n_samples = X.shape[0]
  for i in range(0, n_samples, batch_size):
    yield X[i: min(i + batch_size, n_samples)], y[i: min(i + batch_size, n_samples)]

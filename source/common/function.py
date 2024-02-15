import numpy as np

from copy import deepcopy

class BaseOptimisationFunction:
  def __call__(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
    raise NotImplementedError
  
  def grad(self, w: np.ndarray = None, *args, **kwargs):
    raise NotImplementedError
  
  def update(self, step, *args, **kwargs):
    raise NotImplementedError
  
  def predict(self, X: np.ndarray = None, *args, **kwargs):
    raise NotImplementedError

  def weights(self) -> np.ndarray:
    raise NotImplementedError
  


class MSERidgeLinear(BaseOptimisationFunction):
  def __init__(self, n_features: int, lmbd=0.001):
    # bias always by default
    self.w: np.ndarray = np.ones(shape=(n_features + 1, 1))
    self.lmbd = lmbd
      
  def __call__(self, X: np.ndarray, y: np.ndarray) -> float:
    # self.X.shape = (n_samples, n_features + 1)
    self.X: np.ndarray = np.hstack((X, np.ones((X.shape[0], 1))))
    self.y: np.ndarray = y

    diff = self.X @ self.w - self.y
    return (diff.T @ diff + self.lmbd * self.w.T @ self.w).sum()
  
  def grad(self, w: np.ndarray = None) -> np.ndarray:
    if w is None:
      w = self.w
    return 2 * self.X.T @ (self.X @ w - self.y) + 2 * self.lmbd * w

  def predict(self, X: np.ndarray) -> np.ndarray:
    return np.hstack((X, np.ones((X.shape[0], 1)))) @ self.w

  def update(self, step) -> np.ndarray:
    self.w += step
    
    return self.w

  def weights(self) -> np.ndarray:
    return self.w


class MSELassoLinear(BaseOptimisationFunction):
  def __init__(self, n_features: int, lmbd=0.001):
    # bias always by default
    self.w: np.ndarray = np.ones(shape=(n_features + 1, 1))
    self.lmbd = lmbd
      
  def __call__(self, X: np.ndarray, y: np.ndarray) -> float:
    # self.X.shape = (n_samples, n_features + 1)
    self.X: np.ndarray = np.hstack((X, np.ones((X.shape[0], 1))))
    self.y: np.ndarray = y

    diff = self.X @ self.w - self.y
    return (diff.T @ diff + self.lmbd * np.abs(self.w)).mean()
  
  def grad(self, w: np.ndarray = None) -> np.ndarray:
    if w is None:
      w = self.w
    return 2 * self.X.T @ (self.X @ w - self.y) + self.lmbd * np.sign(w)

  def predict(self, X: np.ndarray) -> np.ndarray:
    return np.hstack((X, np.ones((X.shape[0], 1)))) @ self.w

  def update(self, step) -> np.ndarray:
    self.w += step
    return self.w

  def weights(self) -> np.ndarray:
    return self.w
  
  def _copy(self):
    copy = MSERidgeLinear(0)
    copy.w = deepcopy(self.w)
    copy.lmbd = self.lmbd

    return copy
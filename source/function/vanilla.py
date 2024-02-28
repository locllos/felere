from copy import deepcopy

from .api import BaseOptimisationFunction, np

class MSERidgeLinear(BaseOptimisationFunction):
  def __init__(self, n_features: int, lmbd=0.001):
    # bias always by default
    self.w: np.ndarray = np.ones(shape=(n_features + 1, 1))
    self.lmbd = lmbd
    self.last_gradient = None
      
  def __call__(self, X: np.ndarray, y: np.ndarray, requires_grad=True) -> float:
    # self.X.shape = (n_samples, n_features + 1)
    self.X: np.ndarray = np.hstack((X, np.ones((X.shape[0], 1))))
    self.y: np.ndarray = y

    diff = self.X @ self.w - self.y
    return (diff.T @ diff + self.lmbd * self.w.T @ self.w).sum()
  
  def grad(self) -> np.ndarray:
    self.last_gradient = 2 * self.X.T @ (self.X @ self.w - self.y) + 2 * self.lmbd * self.w

    return self.last_gradient

  def last_grad(self) -> np.ndarray:
    return self.last_gradient

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
  
  def grad(self) -> np.ndarray:
    return 2 * self.X.T @ (self.X @ self.w - self.y) + self.lmbd * np.sign(self.w)

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
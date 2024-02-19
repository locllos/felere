import torch

from copy import deepcopy

from .api import BaseOptimisationFunction

class MSERidgeLinear(BaseOptimisationFunction):
  def __init__(self, n_features: int, lmbd=0.001):
    # bias always by default
    self.w: torch.Tensor = torch.ones(size=(n_features + 1, 1))
    self.lmbd = lmbd
      
  def __call__(self, X: torch.Tensor, y: torch.Tensor) -> float:
    # self.X.shape = (n_samples, n_features + 1)
    self.X: torch.Tensor = torch.hstack((X, torch.ones((X.shape[0], 1))))
    self.y: torch.Tensor = y

    diff = self.X @ self.w - self.y
    return (diff.T @ diff + self.lmbd * self.w.T @ self.w).sum()
  
  def grad(self) -> torch.Tensor:
    return 2 * self.X.T @ (self.X @ self.w - self.y) + 2 * self.lmbd * self.w

  def predict(self, X: torch.Tensor) -> torch.Tensor:
    return torch.hstack((X, torch.ones((X.shape[0], 1)))) @ self.w

  def update(self, step) -> torch.Tensor:
    self.w += step
    
    return self.w

  def weights(self) -> torch.Tensor:
    return self.w


class MSELassoLinear(BaseOptimisationFunction):
  def __init__(self, n_features: int, lmbd=0.001):
    # bias always by default
    self.w: torch.Tensor = torch.ones(shape=(n_features + 1, 1))
    self.lmbd = lmbd
      
  def __call__(self, X: torch.Tensor, y: torch.Tensor) -> float:
    # self.X.shape = (n_samples, n_features + 1)
    self.X: torch.Tensor = torch.hstack((X, torch.ones((X.shape[0], 1))))
    self.y: torch.Tensor = y

    diff = self.X @ self.w - self.y
    return (diff.T @ diff + self.lmbd * torch.abs(self.w)).mean()
  
  def grad(self) -> torch.Tensor:
    return 2 * self.X.T @ (self.X @ self.w - self.y) + self.lmbd * torch.sign(self.w)

  def predict(self, X: torch.Tensor) -> torch.Tensor:
    return torch.hstack((X, torch.ones((X.shape[0], 1)))) @ self.w

  def update(self, step) -> torch.Tensor:
    self.w += step
    return self.w

  def weights(self) -> torch.Tensor:
    return self.w
  
  def _copy(self):
    copy = MSERidgeLinear(0)
    copy.w = deepcopy(self.w)
    copy.lmbd = self.lmbd

    return copy
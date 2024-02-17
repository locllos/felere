from .api import BaseOptimisationFunction, np

import torch
from torch import nn

class PyTorchFunction(BaseOptimisationFunction):
  def __init__(self, module: nn.Module):
    self.module = module

  def __call__(self, X: np.ndarray, y: np.ndarray):
    return self.module.forward(torch.from_numpy(X), torch.from_numpy(y))
  
  def grad(self, w: np.ndarray = None, *args, **kwargs):
    raise NotImplementedError
  
  def update(self, step, *args, **kwargs):
    raise NotImplementedError
  
  def predict(self, X: np.ndarray = None, *args, **kwargs):
    raise NotImplementedError

  def weights(self) -> np.ndarray:
    raise NotImplementedError

  
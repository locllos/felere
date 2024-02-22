import numpy as np

class BaseOptimisationFunction:
  def __call__(self, X: np.ndarray, y: np.ndarray, requires_grad: bool, *args, **kwargs):
    raise NotImplementedError
  
  def grad(self, *args, **kwargs) -> np.ndarray:
    raise NotImplementedError
  
  def update(self, step, *args, **kwargs):
    raise NotImplementedError
  
  def predict(self, X: np.ndarray = None, *args, **kwargs):
    raise NotImplementedError

  def weights(self) -> np.ndarray:
    raise NotImplementedError
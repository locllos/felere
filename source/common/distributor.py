import numpy as np

from .generator import splitter

class BaseDataDistributor:
  def __init__(self):
    raise NotImplementedError
  
  def clients_portions(self):
    raise NotImplementedError
  
  def server_portion(self):
    raise NotImplementedError
  

class UniformDataDistributor(BaseDataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray, 
    n_parts: int, 
    server_fraction: float = 0, 
  ):
    self.X = X
    self.y = y
    self.n_parts = n_parts
    self.server_portion_size = int(server_fraction * self.X.shape[0]) \
                               if server_fraction != 0 else self.X.shape[0] // self.n_parts + 1

  def clients_portions(self):
    return splitter(
      self.X[self.server_portion_size:],
      self.y[self.server_portion_size:], 
      n_parts=self.n_parts
    )
  
  def server_portion(self):
    return self.X[:self.server_portion_size], self.y[:self.server_portion_size]
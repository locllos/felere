import numpy as np

class reducer:
  @staticmethod
  def __call__(array: np.ndarray[np.ndarray]):
    raise NotImplementedError
  
  @staticmethod
  def __repr__():
    return "reducer"


class mean(reducer):
  @staticmethod
  def __call__(array: np.ndarray[np.ndarray]):
    return np.nan_to_num(array, nan=0).mean(axis=0)
  @staticmethod
  def __repr__():
    return "mean"
  

class max(reducer):
  @staticmethod
  def __call__(array: np.ndarray[np.ndarray]):
    return np.nan_to_num(array, nan=-np.inf).max(axis=0)
  
  @staticmethod
  def __repr__():
    return "max"
  
class min(reducer):
  @staticmethod
  def __call__(array: np.ndarray[np.ndarray]):
    return np.nan_to_num(array, nan=+np.inf).min(axis=0)
  
  @staticmethod
  def __repr__():
    return "min"
  

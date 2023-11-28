import numpy as np

class ISampler:
  def __call__():
    raise NotImplementedError
  
class ExponentialSampler(ISampler):
  def __init__(self, scale=1):
    self.scale = scale

  def __call__(self):
    return np.random.exponential(
      scale=self.scale
    )
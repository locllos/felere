from common.sampler import ISampler, ExponentialSampler
from common.decorators import singleton

from common.logging import _logger

@singleton
class Timer:
  def __init__(self, sampler: ISampler = ExponentialSampler()):
    self.time: float = 0
    self.sample = sampler
  
  # every call it tick and return time
  def __call__(self):
    now = self.time 
    self.time += self.sample()

    return now

  def speedup(self, on: float):
    _logger.info(f"Timer was speed up on {on}")
    self.time += on

timer = Timer()
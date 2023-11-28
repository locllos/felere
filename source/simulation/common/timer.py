from common.sampler import ISampler, ExponentialSampler
from common.decorators import singleton


@singleton
class SimulationTimer:
  def __init__(self, sampler: ISampler = ExponentialSampler()):
    self.time: float = 0
    self.sample = sampler
  
  # every call it tick and return time
  def __call__(self):
    now = self.time 
    self.time += self.sample()

    return now

  def speedup(self, on: float):
    self.time += on

simulation_timer = SimulationTimer()
from time import time

class Timer:
  def __init__(self):
    self.clock: int = 0
  
  def __call__(self):
    return time()

  def speedup(on: int):
    pass
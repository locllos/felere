class Differentiable:
  def __call__(self, x):
    raise NotImplementedError
  
  def grad(self, x):
    raise NotImplementedError


class IOptimizer:
  def __init__(self, func: Differentiable):
    self.func = func

  def step(self, batch):
    raise NotImplementedError
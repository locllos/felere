class BaseFederatedOptimizer:
  def __init__(self):
    raise NotImplementedError
  
  def optimize(self, return_history=False):
    raise NotImplementedError
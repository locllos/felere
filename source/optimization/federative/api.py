from common.model import Model

class BaseFederatedOptimizer:
  def __init__(self):
    raise NotImplementedError
  
  def play_round(self, model: Model):
    self._play_round(model)
    model.round_done()

  def _play_round(self, model: Model):
    raise NotImplementedError

  def optimize(self, model: Model, rounds: int):
    for _ in range(rounds):
      self.play_round(model)

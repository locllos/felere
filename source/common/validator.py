from typing import List

from common.distributor import BaseDataDistributor
from common.model import Model


class Validator:
  def __init__(
      self,
      distributor: BaseDataDistributor,
      metrics: List[callable],
      reduce: callable
  ):
    self.metrics: List[callable] = metrics
    self.distributor: BaseDataDistributor = distributor
  
  def validate(self, model: Model):
    scores = {
      "server" : dict(zip(self.metrics, [[] for _ in range(len(self.metrics))])),
      "client" : dict(zip(self.metrics, [[] for _ in range(len(self.metrics))]))
    }



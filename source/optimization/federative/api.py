import numpy as np

from typing import Dict

from common.model import Model

class BaseFederatedOptimizer:
  def __init__(self):
    raise NotImplementedError

  def play_round(self, model: Model):
    raise NotImplementedError
  
  def client_update(
    self,
    server: Model.Agent,
    client: Model.Agent
  ):
    raise NotImplementedError

import ray
import numpy as np

from typing import Dict

from common.simulation import Simulation

class BaseFederatedOptimizer:
  def __init__(self):
    raise NotImplementedError

  def play_round(self, model: Simulation):
    raise NotImplementedError
  
  def client_update(
    self,
    server: Simulation.Agent,
    client: Simulation.Agent
  ):
    raise NotImplementedError
  
  @staticmethod
  def __repr__():
    return "FederatedOptimizer"

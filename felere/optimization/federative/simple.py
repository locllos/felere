from copy import copy
from typing import Dict

import numpy as np

from common.generator import batch_generator

from .api import BaseFederatedOptimizer, Simulation

class Custom(BaseFederatedOptimizer):
  def __init__(self, eta):
    self.eta: float = eta      
    
  def play_round(self, model: Simulation):
    _, clients_weights, other = model.clients_update(self.client_update)
    clients_n_samples = other["n_samples"]
      
    next_global_weights = \
      (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )

  def client_update(self, server, client):
    client.function.update(
      (-1) * (client.function.weights() - server.function.weights())
    )
    client.function(X=client.X, y=client.y)

    step = (-1) * self.eta * client.function.grad()
    client.function.update(step)
    
    client.other["n_samples"] = client.X.shape[0]
    return client
  

  def __repr__(self):
    return "CustomMethod"
from copy import copy
from typing import Dict

import numpy as np

from felere.common.generator import batch_generator

from .api import BaseFederatedOptimizer, Simulation

import ray

class FederatedAveraging(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 16,
    epochs: int = 8,
    eta: float = 1e-3,
  ):
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.eta: float = eta      
    
  def play_round(
    self,
    model: Simulation
  ):
    # make update on clients and get aggregated result
    _, clients_weights, other = model.clients_update(self.client_update)
    clients_n_samples = other["n_samples"]
      
    # global weights update
    next_global_weights = \
      (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )

  def client_update(
    self,
    server: Simulation.Agent,
    client: Simulation.Agent
  ):
    client.function.update(
      (-1) * (client.function.weights() - server.function.weights())
    )
    for _ in range(self.epochs):
      for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
        client.function(X=X_batch, y=y_batch)

        step = (-1) * self.eta * client.function.grad()
        client.function.update(step)
    
    client.other["n_samples"] = client.X.shape[0]
    return client
  

  def __repr__(self):
    return "FederatedAveraging"
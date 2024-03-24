import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
from copy import copy, deepcopy

from function.api import BaseOptimisationFunction
from common.generator import batch_generator
from common.distributor import DataDistributor

from .api import BaseFederatedOptimizer, Simulation


class Scaffold(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 16,
    epochs: int = 8,
    rounds: int = 32,
    eta: float = 1e-3,
    use_grad_for_control = False,
  ):
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.rounds: int = rounds
    self.eta: float = eta
    self.use_grad_for_control = use_grad_for_control

  def play_round(
    self,
    model: Simulation
  ):
    if model.server.other.get("control", None) is None:
      Scaffold._init_controls(model)

    # make update on clients and get aggregated result
    m, clients_weights, other = model.clients_update(self.client_update)
    client_controls_diffs = other["control_diffs"]
  
    # global weights and control update
    next_global_weights = clients_weights.sum(axis=0) / m
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )
    model.server.other["control"] += client_controls_diffs.sum(axis=0) / model.n_clients


  def client_update(self, server: Simulation.Agent, client: Simulation.Agent):
    client.function.update(
      (-1) * (client.function.weights() - server.function.weights())
    )
    for _ in range(self.epochs):
      for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
        client.function(X=X_batch, y=y_batch)

        step = (-1) * self.eta * (client.function.grad() + \
                                  server.other["control"] - client.other["control"])
        client.function.update(step)
    
    next_client_control = np.array([])
    if self.use_grad_for_control:
      next_client_control = client.function.grad()
    else:
      next_client_control = (client.other["control"] - server.other["control"]) + \
                            (server.function.weights() - client.function.weights()) / (self.epochs * self.eta)

    client.other["control_diffs"] = next_client_control - client.other["control"]
    client.other["control"] = next_client_control
    return client
    
  @staticmethod
  def _init_controls(model: Simulation):
    model.server.other["control"] = np.zeros_like(model.server.function.weights())
    client: Simulation.Agent
    for client in model.clients:
      client.other["control"] = np.zeros_like(model.server.function.weights())
  
  def __repr__(self):
    return "Scaffold"
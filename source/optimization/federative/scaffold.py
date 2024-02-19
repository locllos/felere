import torch
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
from copy import copy, deepcopy

from function.api import BaseOptimisationFunction
from common.generator import batch_generator
from common.distributor import BaseDataDistributor

from .api import BaseFederatedOptimizer, Model


class Scaffold(BaseFederatedOptimizer):
  def __init__(
    self,
    clients_fraction: float = 0.3,
    batch_size: int = 16,
    epochs: int = 8,
    rounds: int = 32,
    local_eta: float = 1e-3,
    global_eta: float = 1,
    use_grad_for_control = False,
    return_global_history = False,
  ):
    self.clients_fraction: float = clients_fraction
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.rounds: int = rounds
    self.local_eta: float = local_eta
    self.global_eta: float = global_eta
    self.use_grad_for_control = use_grad_for_control
    self.return_global_history = return_global_history

  def play_round(
    self,
    model: Model
  ):
    if model.server.other.get("control", None) is None:
      Scaffold._init_controls(model)

    m = max(1, int(self.clients_fraction * model.n_clients))
    if model.save_history:
      model.server.history.append(model.server.function(X=model.server.X, y=model.server.y))

    subset = np.random.choice(model.n_clients, m)
    clients_weights: torch.Tensor = torch.zeros((model.n_clients, *model.server.function.weights().shape))
    client_controls_diffs: torch.Tensor = torch.zeros((model.n_clients, *model.server.function.weights().shape))

    client: Model.Agent
    for k, client in zip(subset, model.clients[subset]): # to be optimized: use enumarate to compute weighted weights more efficient
      # client update
      client.function.update(
        (-1) * (client.function.weights() - model.server.function.weights())
      )
      for _ in range(self.epochs):
        for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
          client.history.append(client.function(X=X_batch, y=y_batch))

          step = (-1) * \
                 self.local_eta * \
                 (client.function.grad() + model.server.other["control"] - client.other["control"])
          client.function.update(step)
      
      next_client_control = torch.Tensor([])
      if self.use_grad_for_control:
        next_client_control = client.function.grad()
      else:
        next_client_control = (client.other["control"] - model.server.other["control"]) + \
                              (model.server.function.weights() - client.function.weights()) / (self.epochs * self.local_eta)
      
      # return weights and metadata to the server
      clients_weights[k] = client.function.weights()
      client_controls_diffs[k] = next_client_control - client.other["control"]
      client.other["control"] = next_client_control
  
    # global weights and control update
    next_global_weights = self.global_eta * clients_weights.sum(axis=0) / m
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )
    model.server.other["control"] += client_controls_diffs.sum(axis=0) / model.n_clients


  def _init_controls(model: Model):
    model.server.other["control"] = torch.zeros_like(model.server.function.weights())
    client: Model.Agent
    for client in model.clients:
      client.other["control"] = torch.zeros_like(model.server.function.weights())
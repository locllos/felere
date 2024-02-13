import numpy as np

from dataclasses import dataclass
from typing import List, Tuple
from copy import deepcopy

from common.function import BaseOptimisationFunction
from common.generator import batch_generator
from common.distributor import BaseDataDistributor

from .api import BaseFederatedOptimizer


class Scaffold(BaseFederatedOptimizer):
  def __init__(
    self,
    function: BaseOptimisationFunction, 
    data_distributor: BaseDataDistributor,
  ):
    self.function: BaseOptimisationFunction = function

    self.X_server: np.ndarray = np.array([])
    self.y_server: np.ndarray = np.array([])
    self.clients: List[Scaffold.Client] = np.array([])

    # distribute data to the clients
    for X_portion, y_portion in data_distributor.clients_portions():
      self.clients = np.append(
        self.clients,
        Scaffold.Client(
          X_portion,
          y_portion,
          deepcopy(self.function),
          control=np.zeros_like(self.function.weights())
        )
      )
    
    self.n_clients = len(self.clients)
    self.X_server, self.y_server = data_distributor.server_portion()
      
    
  def optimize(
    self,
    clients_fraction: float = 0.3,
    batch_size: int = 16,
    epochs: int = 8,
    rounds: int = 32,
    eta: float = 1e-3,
    use_grad_for_control = False,
    return_history = False
  ) -> BaseOptimisationFunction | Tuple[BaseOptimisationFunction, List]:
    function = deepcopy(self.function)
    clients = deepcopy(self.clients)

    global_history = []
    m = max(1, int(clients_fraction * self.n_clients))

    global_weights = function.weights()
    global_control = np.zeros_like(global_weights)
    for round in range(rounds):
      global_history.append(function(X=self.X_server, y=self.y_server))

      subset = np.random.choice(self.n_clients, m)
      clients_weights: np.ndarray = np.zeros((self.n_clients, *global_weights.shape))
      client_controls_diffs: np.ndarray = np.zeros((self.n_clients, *global_weights.shape))

      for k, client in zip(subset, clients[subset]): # to be optimized: use enumarate to compute weighted weights more efficient
        # client update
        client.weights = global_weights
        for epoch in range(epochs):
          for X_batch, y_batch in batch_generator(client.X, client.y, batch_size):
            client.history.append(client.function(X=X_batch, y=y_batch))

            step = (-1) * eta * (client.function.grad(client.weights) + \
                                 global_control - client.control)
            client.weights = client.function.update(step)
        
        next_client_control = np.array([])
        if use_grad_for_control:
          next_client_control = client.function.grad(client.weights)
        else:
          next_client_control = (client.control - global_control) + \
                                (global_weights - client.weights) / (epochs * eta)
        
        # return weights and metadata to the server
        clients_weights[k] = client.weights 
        client_controls_diffs[k] = next_client_control - client.control
        client.control = next_client_control
    
      # global weights and control update
      next_global_weights = clients_weights.sum(axis=0) / m
      global_weights = function.update(
        (-1) * (global_weights - next_global_weights)
      )
      global_control += client_controls_diffs.sum(axis=0) / self.n_clients

    if return_history:
      return function, global_history

    return function
  
  @dataclass
  class Client:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    weights: np.ndarray = None
    control: np.ndarray = None
    history = []

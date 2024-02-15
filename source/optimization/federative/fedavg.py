import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
from copy import deepcopy

from common.function import BaseOptimisationFunction
from common.generator import batch_generator
from common.distributor import BaseDataDistributor

from .api import BaseFederatedOptimizer


class FederatedAveraging(BaseFederatedOptimizer):
  def __init__(
    self,
    function: BaseOptimisationFunction, 
    data_distributor: BaseDataDistributor,
  ):
    self.function: BaseOptimisationFunction = function

    self.X_server: np.ndarray = np.array([])
    self.y_server: np.ndarray = np.array([])
    self.clients: np.ndarray[FederatedAveraging.Client] = np.array([])

    # distribute data to the clients
    for X_portion, y_portion in data_distributor.clients_portions():
      self.clients = np.append(
        self.clients,
        FederatedAveraging.Client(X_portion, y_portion, deepcopy(self.function))
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
    return_global_history = False,
  ) -> BaseOptimisationFunction | Tuple[BaseOptimisationFunction, List]:
    function = deepcopy(self.function)

    global_history = []
    m = max(1, int(clients_fraction * self.n_clients))

    global_weights = function.weights()
    for round in range(rounds):
      global_history.append(function(X=self.X_server, y=self.y_server))

      subset = np.random.choice(self.n_clients, m)
      clients_weights: np.ndarray = np.zeros((self.n_clients, *global_weights.shape))
      clients_n_samples: np.ndarray = np.zeros((self.n_clients, *np.ones_like(global_weights.shape)))

      for k, client in zip(subset, self.clients[subset]): # to be optimized: use enumarate to compute weighted weights more efficient
        # client update
        client.weights = global_weights
        for local_epoch in range(epochs):
          for X_batch, y_batch in batch_generator(client.X, client.y, batch_size):
            client.history.append(client.function(X=X_batch, y=y_batch))

            step = (-1) * eta * client.function.grad(client.weights)
            client.weights = client.function.update(step)
        
        # return weights and metadata to the server
        clients_weights[k] = client.weights 
        clients_n_samples[k] = client.X.shape[0]
    
      # global weights update
      next_global_weights = \
        (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
      global_weights = function.update(
        (-1) * (global_weights - next_global_weights)
      )

    if return_global_history:
      return function, global_history

    return function
  
  @dataclass
  class Client:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    weights: np.ndarray = None
    history = []

import numpy as np

from dataclasses import dataclass
from typing import List
from copy import copy

from common.function import BaseOptimisationFunction
from common.generator import splitter, batch_generator
from optimization.single import draw_history

class FederatedAveraging:
  @dataclass
  class Client:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    weights: np.ndarray = None
    history = []

  def __init__(
    self,
    function: BaseOptimisationFunction, 
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 50
  ):
    self.function: BaseOptimisationFunction = function
    self.X: np.ndarray = X
    self.y: np.ndarray = y
    self.n_clients: int = n_clients

    self.X_server: np.ndarray = np.array([])
    self.y_server: np.ndarray = np.array([])
    self.clients: List[FederatedAveraging.Client] = np.array([])

    # distribute data to the clients
    for X_part, y_part in splitter(X, y, n_parts=self.n_clients + 1):
      if self.X_server.shape == (0,) or self.y_server.shape == (0,):
        self.X_server = X_part
        self.y_server = y_part
      else:
        self.clients = np.append(
          self.clients,
          FederatedAveraging.Client(X_part, y_part, copy(self.function))
        )
      
    
  def optimize(
    self,
    clients_fraction: float = 0.3,
    batch_size: int = 16,
    local_epochs: int = 8,
    global_epochs: int = 32,
    eta: float = 1e-3,
    show_history = False
  ):
    global_history = []
    m = max(1, int(clients_fraction * self.n_clients))

    global_weights = self.function.weights()
    for global_epoch in range(global_epochs):
      global_history.append(self.function(X=self.X_server, y=self.y_server))

      subset = np.random.choice(self.n_clients, m)
      clients_weights: np.ndarray = np.zeros((self.n_clients, *global_weights.shape))
      clients_n_samples: np.ndarray = np.zeros((self.n_clients, *np.ones_like(global_weights.shape)))

      for k, client in zip(subset, self.clients[subset]):
        # client update
        client.weights = global_weights
        for local_epoch in range(local_epochs):
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
      global_weights = self.function.update(
        (-1) * (global_weights - next_global_weights)
      )

    if show_history:
      draw_history(global_history)

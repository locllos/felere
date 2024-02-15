import numpy as np


from common.generator import batch_generator

from .api import BaseFederatedOptimizer, Model


class FederatedAveraging(BaseFederatedOptimizer):
  def __init__(
    self,
    clients_fraction: float = 0.3,
    batch_size: int = 16,
    epochs: int = 8,
    eta: float = 1e-3,
  ):
    self.clients_fraction = clients_fraction
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.eta: float = eta      
    
  def play_round(
    self,
    model: Model
  ):
    m = max(1, int(self.clients_fraction * model.n_clients))

    global_weights = model.server.weights
    if model.save_history:
      model.server.history.append(model.server.function(X=model.server.X, y=model.server.y))
    else:
      model.server.function(X=model.server.X, y=model.server.y)

    subset = np.random.choice(model.n_clients, m)
    clients_weights: np.ndarray = np.zeros((model.n_clients, *global_weights.shape))
    clients_n_samples: np.ndarray = np.zeros((model.n_clients, *np.ones_like(global_weights.shape)))

    client: Model.Agent
    for k, client in zip(subset, model.clients[subset]): # to be optimized: use enumarate to compute weighted weights more efficient
      # client update
      client.weights = global_weights
      for local_epoch in range(self.epochs):
        for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
          if model.save_history:
            client.history.append(client.function(X=X_batch, y=y_batch))
          else:
            client.function(X=X_batch, y=y_batch)

          step = (-1) * self.eta * client.function.grad(client.weights)
          client.weights = client.function.update(step)
      
      # return weights and metadata to the server
      clients_weights[k] = client.weights 
      clients_n_samples[k] = client.X.shape[0]
  
    # global weights update
    next_global_weights = \
      (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    
    global_weights = model.server.function.update(
      (-1) * (global_weights - next_global_weights)
    )

  def optimize(self, model: Model, rounds: int):
    for round in range(rounds):
      self.play_round(model)

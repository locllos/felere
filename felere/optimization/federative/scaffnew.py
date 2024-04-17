import numpy as np

from .api import BaseFederatedOptimizer, Simulation
from felere.common.generator import batch_generator

class Scaffnew(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 128,
    eta: float = 1e-3,
    proba: float = 1/512,

  ):
    self.batch_size: int = batch_size
    self.eta: float = eta
    self.max_skip_rounds: float = 1 / proba

  def play_round(self, model: Simulation):
    if model.server.other.get("control", None) is None:
      Scaffnew._init_controls(model)

    m, clients_weights, _ = model.clients_update(self.client_update)
    model.server.function.update(
      (-1) * (model.server.function.weights() - clients_weights.sum(axis=0) / m)
    )
  
  def client_update(self, server: Simulation.Agent, client: Simulation.Agent):
    weights_cap: np.ndarray = client.function.weights()
    weights: np.ndarray = server.function.weights()

    client.other["control"] = client.other["control"] + \
      (weights - weights_cap) / (self.eta * self.max_skip_rounds)
    client.function.update(
      (-1) * (weights_cap - weights)
    )

    for _ in range(np.random.randint(1, self.max_skip_rounds)):
      for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
        client.function(X=X_batch, y=y_batch)

        client.function.update(
          (-1) * self.eta * (client.function.grad() - client.other["control"])
        )

    return client
  
  @staticmethod
  def _init_controls(model: Simulation):
    model.server.other["control"] = np.zeros_like(model.server.function.weights())
    client: Simulation.Agent
    for client in model.clients:
      client.other["control"] = np.zeros_like(model.server.function.weights())    

  def __repr__(self):
    return "Scaffnew"
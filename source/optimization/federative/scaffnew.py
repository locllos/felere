import numpy as np

from .api import BaseFederatedOptimizer, Model

class Scaffnew(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 128,
    epochs: int = 8,
    eta: float = 1e-3,
    proba: float = 0.01
  ):
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.eta: float = eta
    self.proba: float = proba

  def play_round(self, model: Model):
    if model.server.other.get("control", None) is None:
      Scaffnew._init_controls(model)

    for epoch in range(self.epochs):
      model.server.other["skip"] = np.random.choice([True, False], size=1, p=[1-self.proba, self.proba])
      m, clients_weights, _ = model.clients_update(self.client_update)

      model.server.function.update(
        (-1) * (model.server.function.weights() - clients_weights.sum(axis=0) / m)
      )
  
  def client_update(self, server: Model.Agent, client: Model.Agent):
    if server.other["skip"]:
      client.function(X=client.X, y=client.y)

      client.function.update(
        (-1) * self.eta * (client.function.grad() - client.other["control"])
      )
      return client
    
    weights = server.function.weights()
    weights_cap: np.ndarray = client.function.weights()
    client.other["control"] = client.other["control"] + \
      (self.proba / self.eta) * (weights - weights_cap)
    
    client.function.update(
      (-1) * self.eta * (weights_cap - weights)
    )

    return client
    
  
  
  @staticmethod
  def _init_controls(model: Model):
    model.server.other["control"] = np.zeros_like(model.server.function.weights())
    client: Model.Agent
    for client in model.clients:
      client.other["control"] = np.zeros_like(model.server.function.weights())    

import numpy as np

from .api import BaseFederatedOptimizer, Model

class Scaffnew(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 128,
    eta: float = 1e-3,
    epochs: int = 64,
    proba: float = 0.01,

  ):
    self.batch_size: int = batch_size
    self.eta: float = eta
    self.epochs = epochs
    self.communication_rounds: int = int(epochs * proba)
    self.proba: float = proba

  def play_round(self, model: Model):
    if model.server.other.get("control", None) is None:
      Scaffnew._init_controls(model)


    commucations = np.random.choice(
      np.arange(0, self.epochs), size=self.communication_rounds
    )
    prev = 0
    # instead of loop use max_skip_rounds with proba parameter
    for skip_epochs in commucations:
      model.server.other["skip_epochs"] = skip_epochs - prev + 1

      m, clients_weights, _ = model.clients_update(self.client_update)
      model.server.function.update(
        (-1) * (model.server.function.weights() - clients_weights.sum(axis=0) / m)
      )
  
  def client_update(self, server: Model.Agent, client: Model.Agent):
    weights_cap: np.ndarray = client.function.weights()
    weights: np.ndarray = server.function.weights()

    client.other["control"] = client.other["control"] + \
      (self.proba / self.eta) * (weights - weights_cap)
    client.function.update(
      (-1) * (weights_cap - weights)
    )

    for _ in range(server.other["skip_epochs"]):
      client.function(X=client.X, y=client.y)

      client.function.update(
        (-1) * self.eta * (client.function.grad() - client.other["control"])
      )

    return client
  
  @staticmethod
  def _init_controls(model: Model):
    model.server.other["control"] = np.zeros_like(model.server.function.weights())
    client: Model.Agent
    for client in model.clients:
      client.other["control"] = np.zeros_like(model.server.function.weights())    

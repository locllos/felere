from .api import BaseFederatedOptimizer, Simulation

from common.generator import batch_generator


class FedFair(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 16,
    epochs: int = 8,
    eta: float = 1e-3,
    lmbd: float = 4
  ):
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.eta: float = eta  
    self.lmbd: float = lmbd

  def play_round(self, model: Simulation):
    if "loss" not in model.server.other.keys():
      model.server.other["loss"] = model.server.function(
        model.server.X, model.server.y
      )

    _, clients_weights, other = model.clients_update(self.client_update)
    clients_n_samples = other["n_samples"]
    clients_loss = other["loss"]

    next_global_weights = \
      (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )

    model.server.other["loss"] = \
      (clients_loss * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    

    
  def client_update(self, server: Simulation.Agent, client: Simulation.Agent):
    client.function.update(
      (-1) * (client.function.weights() - server.function.weights())
    )
    for _ in range(self.epochs):
      for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
        loss = client.function(X=X_batch, y=y_batch)
        delta = loss - server.other["loss"]

        local_eta = self.eta * (1 + self.lmbd * delta)
        step = (-1) * local_eta * client.function.grad()
        client.function.update(step)
    
    client.other["n_samples"] = client.X.shape[0]
    client.other["loss"] = client.function(client.X, client.y)
    return client
        

  def __repr__(self):
    return "FedFair"
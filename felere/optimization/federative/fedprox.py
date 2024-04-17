from optimization.federative.api import BaseFederatedOptimizer, Simulation

from common.generator import batch_generator

class FedProx(BaseFederatedOptimizer):
  def __init__(
    self,
    batch_size: int = 16,
    epochs: int = 8,
    rounds: int = 32,
    eta: float = 1e-3,
    mu: float = 0.5
  ):
    self.batch_size: int = batch_size
    self.epochs: int = epochs
    self.rounds: int = rounds
    self.eta: float = eta
    self.mu: float = mu
  
  def play_round(
    self,
    model: Simulation
  ):
    # make update on clients and get aggregated result
    m, client_weights, _ = model.clients_update(self.client_update)
    
    # global weights update
    model.server.function.update(
      (-1) * (model.server.function.weights() - client_weights.sum(axis=0) / m)
    )


  def client_update(
    self,
    server: Simulation.Agent,
    client: Simulation.Agent
  ):
    server_weights = server.function.weights()
    client.function.update(
      (-1) * (client.function.weights() - server_weights)
    )
    for _ in range(self.epochs):
      for X_batch, y_batch in batch_generator(client.X, client.y, self.batch_size):
        client.function(X=X_batch, y=y_batch)

        step = (-1) * self.eta * \
          (client.function.grad() + self.mu * (client.function.weights() - server_weights)) # proximal term
       
        client.function.update(step)

    return client
  

  def __repr__(self):
    return "FedProx"
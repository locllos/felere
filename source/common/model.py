import numpy as np

from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy

from function.api import BaseOptimisationFunction
from common.distributor import DataDistributor


class Model:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    X: Dict[str, np.ndarray | List[np.ndarray]],
    y: Dict[str, np.ndarray | List[np.ndarray]]
  ):
    self.clients: np.ndarray[Model.Agent] = np.array([])

    for client_id, (X_portion, y_portion) in enumerate(zip(X["clients"], y["clients"])):
      self.clients = np.append(
        self.clients,
        Model.Agent(
          id=client_id,
          X=X_portion, 
          y=y_portion,
          function=deepcopy(function),
          other=dict()
        )
      )

    self.n_clients = len(self.clients)

    self.server: Model.Agent = Model.Agent(
      id=self.n_clients,
      X=X["server"],
      y=y["server"],
      function=deepcopy(function),
      other=dict()
    )

  def function(self, with_clients=False):
    if not with_clients:
      return self.server.function
    
    client_functions = []

    client: Model.Agent
    for client in self.clients:
      client_functions.append(client.function)

    return self.server.function, client_functions

  def validate(
    self,
    # metric: callable,
    X_val: Dict[str, np.ndarray | List[np.ndarray]],
    y_val: Dict[str, np.ndarray | List[np.ndarray]]
  ):
    return {
      "server"  : self.server.function(X=X_val["server"], y=y_val["server"], requires_grad=False),
      "clients" : np.array([
        client.function(X=X_val["clients"][client.id], y=y_val["clients"][client.id], requires_grad=False)
        for client in self.clients
      ]).reshape(self.n_clients, 1)
    }

  @dataclass
  class Agent:
    id: int
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    other: Dict = None
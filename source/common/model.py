import numpy as np

from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy

from function.api import BaseOptimisationFunction
from common.distributor import BaseDataDistributor

class Model:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    distributor: BaseDataDistributor,
  ):
    self.server: Model.Agent = Model.Agent(
      *distributor.server_portion(),
      function=deepcopy(function),
      history=list(),
      other=dict()
    )

    self.clients: np.ndarray[Model.Agent] = np.array([])
    for X_portion, y_portion in distributor.clients_portions():
      self.clients = np.append(
        self.clients,
        Model.Agent(
          X=X_portion,
          y=y_portion,
          function=deepcopy(function),
          history=None,
          other=dict()
        )
      )
    self.n_clients: int = len(self.clients)

  def function(self, with_clients=False):
    if not with_clients:
      return self.server.function
    
    client_functions = []

    client: Model.Agent
    for client in self.clients:
      client_functions.append(client.function)

    return self.server.function, client_functions

  @dataclass
  class Agent:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    other: Dict = None

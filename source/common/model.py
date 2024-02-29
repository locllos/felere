import numpy as np

from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy

from function.api import BaseOptimisationFunction
from common.distributor import DataDistributor
from concurrent.futures import Executor, wait


class Model:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    X: Dict[str, np.ndarray | List[np.ndarray]],
    y: Dict[str, np.ndarray | List[np.ndarray]],
    clients_fraction: float = 0.3,
    executor: Executor = None
  ):
    self.clients: np.ndarray[Model.Agent] = np.array([])
    self.clients_fraction: float = clients_fraction
    self.executor: Executor = executor

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

  def clients_update(
    self,
    update_function: callable
  ):
    m = max(1, int(self.clients_fraction * self.n_clients))
    subset = np.random.choice(self.n_clients, m, replace=False)

    if self.executor is None:
      for client_id in subset:
        update_function(self.server, self.clients[client_id])
    
    else:
      _, failed = wait([
        self.executor.submit(
          update_function, model=self, client=self.clients[client_id]
        )
        for client_id in subset
      ], return_when="ALL_COMPLETED")

      if len(failed) > 0:
        print(failed)
        raise ValueError

    client: Model.Agent
    weights: np.ndarray[np.ndarray] = None
    other: Dict[str, np.ndarray] = {}

    for client in self.clients[subset]:
      if weights is None:
        weights = client.function.weights()
      else:
        weights = np.vstack([weights, client.function.weights()])

      for key, item in client.other.items():
        if key not in other.keys():
          other[key] = item
        else:
          other[key] = np.vstack([other[key], item])
          

    return m, weights, other

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
    y_val: Dict[str, np.ndarray | List[np.ndarray]],
    metrics: Dict[str, callable] = {}
  ):
    results = {
      "server"  : self.server.function(X=X_val["server"], y=y_val["server"], requires_grad=True),
      "clients" : np.array([
        client.function(X=X_val["clients"][client.id], y=y_val["clients"][client.id], requires_grad=False)
        for client in self.clients
      ]).reshape(self.n_clients, 1),
      "norm_grad" : np.linalg.norm(self.server.function.grad(), ord=2) 
    }
    if len(metrics) == 0:
      return results
    
    results["metrics"] = {}
    for name, metric in metrics.items():
      results["metrics"][name] = metric(self.server.function.predict(X_val["server"]), y_val["server"])

    return results

  @dataclass
  class Agent:
    id: int
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    other: Dict = None
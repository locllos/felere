import numpy as np

from dataclasses import dataclass
from typing import Dict, List
from copy import deepcopy

from common.function import BaseOptimisationFunction
from common.distributor import BaseDataDistributor

class Model:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    distributor: BaseDataDistributor,
    save_history: bool = False
  ):
    self.function: BaseOptimisationFunction = function

    self.server: Model.Agent = Model.Agent(
      *distributor.server_portion(),
      deepcopy(function),
    )
    self.clients: np.ndarray[Model.Agent] = np.array([])
    for X_portion, y_portion in distributor.clients_portions():
      self.clients = np.append(
        self.clients,
        Model.Agent(X_portion, y_portion, deepcopy(self.function))
      )
    self.n_clients: int = len(self.clients)
    self.save_history: bool = save_history

  @dataclass
  class Agent:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    weights: np.ndarray = None    
    history: List = []
    other: Dict = None
    
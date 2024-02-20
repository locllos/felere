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
    save_history: bool = False
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
    self.save_history: bool = save_history
    
    if self.save_history:
      self.client_history_manager = ClientHistoryManager(self.n_clients)
      
      client: Model.Agent
      for client in self.clients:
        client.history = self.client_history_manager.create()

  def function(self, with_clients=False):
    if not with_clients:
      return self.server.function
    
    client_functions = []

    client: Model.Agent
    for client in self.clients:
      client_functions.append(client.function)

    return self.server.function, client_functions
  
  def round_done(self):
    self.client_history_manager.prepare_new_round()


  @dataclass
  class Agent:
    X: np.ndarray
    y: np.ndarray
    function: BaseOptimisationFunction
    # weights: np.ndarray = None
    history: list = None
    other: Dict = None


class ClientHistoryManager:
  def __init__(self, n_clients: int):
    self.n_clients = n_clients
    self.current_id = 0

    self.histories: np.ndarray = np.full(shape=(self.n_clients, 1), fill_value=np.nan)

  def create(self):
    return ClientHistoryManager.ClientHistory(self)

  def mean(self):
    return np.nan_to_num(self.histories, nan=0).mean(axis=0)

  def max(self):
    return np.nan_to_num(self.histories, nan=-np.inf).max(axis=0)

  def min(self):
    return np.nan_to_num(self.histories, nan=+np.inf).min(axis=0)

  def sum(self):
    return np.nan_to_num(self.histories, nan=0).sum(axis=0)


  def _next_id(self):
    if (self.current_id == self.n_clients):
      raise BufferError
    
    current = self.current_id
    self.current_id += 1

    return current

  def _append(self, history, value):
    self.histories[history.id][-1] = value
  
  def prepare_new_round(self):
    self.histories = np.hstack((
      self.histories, np.full(shape=(self.n_clients, 1), fill_value=np.nan)
    ))

  class ClientHistory:
    def __init__(self, manager):
      self.manager = manager
      self.id = manager._next_id()

    def append(self, value: float):
      self.manager._append(self, value)
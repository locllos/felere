import numpy as np

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
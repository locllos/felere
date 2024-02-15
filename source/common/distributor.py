import numpy as np

from .generator import splitter

class BaseDataDistributor:
  def __init__(self):
    raise NotImplementedError
  
  def clients_portions(self):
    raise NotImplementedError
  
  def server_portion(self):
    raise NotImplementedError
  

class UniformDataDistributor(BaseDataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray, 
    n_parts: int, 
    server_fraction: float = 0, 
  ):
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)

    self.X = X[shuffled_indices]
    self.y = y[shuffled_indices]
    self.n_parts = n_parts
    self.server_portion_size = int(server_fraction * self.X.shape[0]) \
                               if server_fraction != 0 else self.X.shape[0] // (self.n_parts + 1)

  def clients_portions(self):
    return splitter(
      self.X[self.server_portion_size:],
      self.y[self.server_portion_size:], 
      n_parts=self.n_parts
    )
  
  def server_portion(self):
    return self.X[:self.server_portion_size], self.y[:self.server_portion_size]
  
class HomogenousDataDistributor(BaseDataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    n_parts: int,
    iid_fraction: float = 0.3,
    server_fraction: float = 0  
  ):
    self.iid_fraction = iid_fraction

    indices = np.arange(0, y.shape[0])
    np.random.shuffle(indices)

    iid = indices[:self._iid_size(len(indices))]
    self.X_iid = X[iid]
    self.y_iid = y[iid]
    
    non_iid = indices[self._iid_size(len(indices)):]
    self.X_non_iid = X[non_iid]
    self.y_non_iid = y[non_iid]

    sorted_non_iid = np.argsort(self._by_norm1(self.y_non_iid))
    self.X_non_iid = self.X_non_iid[sorted_non_iid]
    self.y_non_iid = self.y_non_iid[sorted_non_iid]

    server_size = int(server_fraction * y.shape[0]) \
                  if server_fraction != 0 else y.shape[0] // (n_parts + 1)
    self.server_iid_size = self._iid_size(server_size)
    self.server_non_iid_size = server_size - self.server_iid_size

    client_size = (y.shape[0] - server_size) // n_parts + 1
    self.client_iid_size = self._iid_size(client_size)
    self.client_non_iid_size = client_size - self.server_iid_size


  def clients_portions(self):
    return self._client_splitter()

  def server_portion(self):
    return self._pack_data(
      iid_from=0, iid_to=self.server_iid_size, 
      non_iid_from=0, non_iid_to=self.server_non_iid_size
    )

  def _pack_data(
    self,
    iid_from, iid_to, 
    non_iid_from, non_iid_to
  ):
    return np.vstack((self.X_iid[iid_from : iid_to], self.X_non_iid[non_iid_from : non_iid_to])), \
           np.vstack((self.y_iid[iid_from : iid_to], self.y_non_iid[non_iid_from : non_iid_to])),

  def _iid_size(self, total: int):
    return int(self.iid_fraction * total)
  
  def _by_norm1(self, data: np.ndarray):
    return np.linalg.norm(data + abs(data.min()), ord=1, axis=1)
  
  def _client_splitter(self):
    iid_indices = \
      range(self.server_iid_size, self.y_iid.shape[0], self.client_iid_size)
    non_iid_indices = \
      range(self.server_non_iid_size, self.y_non_iid.shape[0], self.client_non_iid_size)
    packed_range = zip(iid_indices, non_iid_indices)

    for iid_from, non_iid_from in packed_range:
      yield self._pack_data(
        iid_from=iid_from,
        iid_to=min(iid_from + self.client_iid_size, self.y_iid.shape[0]),
        non_iid_from=non_iid_from,
        non_iid_to=min(non_iid_from + self.client_non_iid_size, self.y_non_iid.shape[0]),
      )


class HomogenousDataDistributorComplex(BaseDataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    n_parts: int,
    iid_fraction: float = 0.3,
    server_fraction: float = 0  
  ):
      
    self.non_iid_order: np.ndarray = np.argsort(self._by_norm1(y))
    self.X: np.ndarray = X
    self.y: np.ndarray = y
    server_size = int(server_fraction * self.X.shape[0]) \
                  if server_fraction != 0 else self.X.shape[0] // (n_parts + 1) + 1
    
    self.server_X, self.server_y = self._extract_data(
      int(iid_fraction * server_size),
      server_size - int(iid_fraction * server_size)
    )

    client_size = (self.X.shape[0] - server_size) // n_parts + 1
    self.iid_size = \
      int(iid_fraction * client_size)
    self.homogenous_size = client_size - self.iid_size


  def clients_portions(self):
    return self._splitter()

  def server_portion(self):
    return self.server_X, self.server_y

  def _extract_data(self, iid_size, non_iid_size):
      # sampling iid 
      iid_part = np.random.choice(
        self.non_iid_order, min(iid_size, len(self.non_iid_order)), replace=False
      )
      # removing already used part
      self.non_iid_order = np.delete(
        self.non_iid_order,
        np.argwhere(np.isin(self.non_iid_order, iid_part) == True).T[0]
      )
      # extracting non iid
      non_iid_part = self.non_iid_order[
        :min(non_iid_size, self.non_iid_order.shape[0])
      ]
      # removing already used part
      self.non_iid_order = np.delete(
        self.non_iid_order, 
        range(min(non_iid_size, self.non_iid_order.shape[0]))
      )
      
      return np.vstack((self.X[iid_part], self.X[non_iid_part])), \
             np.vstack((self.y[iid_part], self.y[non_iid_part]))

  def _by_norm1(self, data: np.ndarray):
    return np.linalg.norm(data + abs(data.min()), ord=1, axis=1)
  
  def _splitter(self):
    while self.non_iid_order.size > 0:
      yield self._extract_data(
        self.iid_size, self.homogenous_size
      )
import numpy as np

from .generator import splitter
from sklearn.model_selection import train_test_split


class BaseDataDistributor:
  def __init__(self):
    raise NotImplementedError
  
  def distribute(
    self, 
    X: np.ndarray,
    y: np.ndarray,
    n_parts: int,
    test_size: float = 0
  ):
    raise NotImplementedError

# distributor should distribute not at once, but on demand
# method `distribute` we need that will return distributed data for n_clients
# distributer implements only strategy of distribution

class DataDistributor:
  def __init__(
    self,
    server_fraction: float = 0,
    test_size: float = 0,
  ):
    self.server_fraction: float = server_fraction
    self.test_size: float = test_size

  def distribute(
    self, 
    X: np.ndarray,
    y: np.ndarray,
    n_parts: int,
    iid_fraction: float = 0.3,
  ):
    """
    returns dict like:
      "train" : {
          "X" : {
            "server" : np.ndarray,
            "clients" : [np.ndarray] 
          },
          "y" : {
            "server" : np.ndarray,
            "clients" : [np.ndarray] 
          }
        }
    """
    self.non_iid_order: np.ndarray = np.argsort(self._by_norm1(y))
    if X.dtype in [np.int32, np.int64]:
      self.X = np.array(X, dtype=np.int64)
    elif X.dtype in [np.float32, np.float64]:
      self.X = np.array(X, dtype=np.float32)
    else:
      raise TypeError
    if y.dtype in [np.int32, np.int64]:
      self.y = np.array(y, dtype=np.int64)
    elif y.dtype in [np.float32, np.float64]:
      self.y = np.array(y, dtype=np.float32)
    else:
      raise TypeError

    server_size = int(self.server_fraction * self.X.shape[0]) \
                  if self.server_fraction != 0 else self.X.shape[0] // (n_parts + 1) + 1
    
    self.server_X, self.server_y = self._extract_data(server_size, 0)

    client_size = (self.X.shape[0] - server_size) // n_parts + 1
    self.iid_size = \
      int(iid_fraction * client_size)
    self.homogenous_size = client_size - self.iid_size

    # consider take out to base class this 
    data = {
      "server" : {"X" : self.server_X, "y" : self.server_y},
      "clients" : [{"X" : X, "y" : y} for X, y in self._splitter()]
    }

    if self.test_size == 0:
      return self._train_split(data)

    return self._train_test_split(data)

  def _train_split(self, data):
    splitted_data = {
      "train" : {
        "X" : {
          "server" : None,
          "clients" : [] 
        },
        "y" : {
          "server" : None,
          "clients" : [] 
        }
      }
    }
    splitted_data["train"]["X"]["server"] = data["server"]["X"]
    splitted_data["train"]["y"]["server"] = data["server"]["y"]

    for client_data in data["clients"]:
      splitted_data["train"]["X"]["clients"].append(client_data["X"])
      splitted_data["train"]["y"]["clients"].append(client_data["y"])

    return splitted_data


  def _train_test_split(self, data):
    splitted_data = {
      "train" : {
        "X" : {
          "server" : None,
          "clients" : [] 
        },
        "y" : {
          "server" : None,
          "clients" : [] 
        }
      },
      "test" : {
        "X" : {
          "server" : None,
          "clients" : [] 
        },
        "y" : {
          "server" : None,
          "clients" : [] 
        }
      }
    }
    self._split_for_server(splitted_data, data)
    self._split_for_client(splitted_data, data)

    return splitted_data
    
  def _split_for_server(self, splitted_data, data):
    X_train, X_test, y_train, y_test = train_test_split(
      data["server"]["X"], data["server"]["y"], test_size=self.test_size, random_state=42
    )
    splitted_data["train"]["X"]["server"] = X_train
    splitted_data["train"]["y"]["server"] = y_train

    splitted_data["test"]["X"]["server"] = X_test
    splitted_data["test"]["y"]["server"] = y_test
                        
  def _split_for_client(self, splitted_data, data):
    for client_data in data["clients"]:
      X_train, X_test, y_train, y_test = train_test_split(
        client_data["X"], client_data["y"], test_size=self.test_size, random_state=42
      )
      splitted_data["train"]["X"]["clients"].append(X_train)
      splitted_data["train"]["y"]["clients"].append(y_train)

      splitted_data["test"]["X"]["clients"].append(X_test)
      splitted_data["test"]["y"]["clients"].append(y_test)
                        

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
      if (len(non_iid_part) == 0):
        return self.X[iid_part], self.y[iid_part]
      
      if (len(iid_part) == 0):
        return self.X[non_iid_part], self.y[non_iid_part]
      
      if (len(self.y.shape) == 1):
        return np.vstack((self.X[iid_part], self.X[non_iid_part])), \
               np.append(self.y[iid_part], self.y[non_iid_part])
      
      return np.vstack((self.X[iid_part], self.X[non_iid_part])), \
              np.vstack((self.y[iid_part], self.y[non_iid_part]))
        


  def _by_norm1(self, data: np.ndarray):
    vector = data

    if len(vector.shape) == 1:
      vector = data.reshape((data.shape[0], 1))

    return np.linalg.norm(vector + abs(vector.min()), ord=1, axis=1)
  
  def _splitter(self):
    while self.non_iid_order.size > 0:
      yield self._extract_data(
        self.iid_size, self.homogenous_size
      )











class UniformDataDistributor(DataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray, 
    n_parts: int, 
    server_fraction: float = 0, 
  ):
    raise NotImplementedError
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
  
class HomogenousDataDistributor(DataDistributor):
  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    n_parts: int,
    iid_fraction: float = 0.3,
    server_fraction: float = 0  
  ):
    raise NotImplementedError
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

from typing import Dict, List
import numpy as np

class HistoryManager:
  def __init__(self):
    self.history: Dict[str, Dict[str, np.ndarray]] = {}
    
  def mean_locals(self, key: str):
    return np.nan_to_num(self.history.get(key, np.array([]))["clients"], nan=0).mean(axis=0)

  def max_locals(self, key):
    return np.nan_to_num(self.history.get(key, np.array([]))["clients"], nan=-np.inf).max(axis=0)

  def min_locals(self, key):
    return np.nan_to_num(self.history.get(key, np.array([]))["clients"], nan=+np.inf).min(axis=0)

  def sum_locals(self, key):
    return np.nan_to_num(self.history.get(key, np.array([]))["clients"], nan=0).sum(axis=0)

  def append(self, key, results):
    if key not in self.history.keys():
      self.history[key] = {
        "server"  : np.array([results["server"]]),
        "clients" :  results["clients"],
        "norm_grads" : results["norm_grad"] 
                       if results["norm_grad"] is not None else np.array([]),
        "metrics" : results["metrics"]
      }
    else:
      self.history[key]["server"] = np.append(self.history[key]["server"], results["server"])
      self.history[key]["clients"] = np.hstack((self.history[key]["clients"], results["clients"]))
      self.history[key]["norm_grads"] = np.append(self.history[key]["norm_grads"],  results["norm_grad"])

      for metric, result in results["metrics"].items():
        self.history[key]["metrics"][metric] = np.append(self.history[key]["metrics"][metric], result)

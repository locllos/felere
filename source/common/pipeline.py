import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from function.api import BaseOptimisationFunction
from optimization.federative.fedavg import BaseFederatedOptimizer
from common.model import Model

from itertools import product
from functools import reduce
from typing import Dict, List, Tuple, Type
from scipy.interpolate import make_interp_spline

class Pipeline:
  def __init__(
    self,
    model: Model,
    optimizer: Type,
    metrics: Dict[str, callable],
    parameters: Dict[str, List], 
    X_val: np.ndarray, 
    y_val: np.ndarray
  ):
    self.model = model
    self.optimizer = optimizer
    self.metrics: Dict[callable] = metrics

    self.parameters_keys = parameters.keys()
    self.parameters_lists: List[List] = list(product(*parameters.values()))
    self.parameters_lists_count = len(self.parameters_lists)

    self.X_val = X_val
    self.y_val = y_val
  
  def run(
      self,
      choose_best_by,
      show_history: bool = False,
      reduce_clients_by: List[str] = None
    ) -> Tuple[Model, Dict[str, List]]:
    best_model: Model = None
    best_metric_value = np.inf
    best_parameters = {}

    fig: plt.Figure = None
    axes: plt.Axes = None
    if show_history:
      num_columns = 1 if reduce_clients_by is None else len(reduce_clients_by) + 1
      fig, axes =  plt.subplots(
        len(self.parameters_lists), num_columns, figsize=(7.5 * num_columns, self.parameters_lists_count * 2.5)
      )
      if type(axes) is not np.ndarray or \
         len(self.parameters_lists) == 1 and len(axes) == num_columns:
        axes = np.array([axes]) 
      
    for i, parameters_list in enumerate(self.parameters_lists):
      parameters = dict(zip(self.parameters_keys, parameters_list))
      rounds = parameters.pop("rounds", 1)

      optimizer: BaseFederatedOptimizer = self.optimizer(**parameters)
      model = deepcopy(self.model)
      
      optimizer.optimize(model, rounds)
      if show_history and reduce_clients_by is None:
          self._draw_global_history(axes[i], model.server.history, parameters)
      elif show_history and reduce_clients_by is not None:
          self._draw_all_history(
            axes[i],
            model.server.history,
            self._reduce_client_history(model, parameters.get("clients_fraction", 1), reduce_by=reduce_clients_by),
            parameters,
            reduce_clients_by
          )

        
      print(f"\nFor parameters: {parameters}:")
      for key, metric in self.metrics.items():
        computed_metric = metric(self.y_val, model.server.function.predict(self.X_val))
        if key == choose_best_by and computed_metric < best_metric_value:
          best_metric_value = computed_metric
          best_parameters = parameters
          best_model = model

        print(f"{key} : {computed_metric}")

    if show_history:
      fig.tight_layout()

    return best_model, best_parameters


  def _draw_global_history(
    self,
    axes: plt.Axes,
    global_history: List,
    parameters: Dict
  ):
    axes.plot(np.arange(1, len(global_history) + 1), global_history)
    axes.set_xlabel("steps")
    axes.set_ylabel("function value")
    axes.set_title(f"{parameters}")

  def _draw_all_history(
      self,
      axes: np.ndarray[plt.Axes], 
      global_history: List, 
      local_history: np.ndarray,
      parameters: Dict,
      reduced_by: List[str]
    ):
    
    axes[0].plot(np.arange(1, len(global_history) + 1), global_history)
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("function value")
    axes[0].set_title(f"global with {parameters}")

    for i, ax in enumerate(axes[1:]):
      ax.plot(np.arange(1, len(local_history[i]) + 1), local_history[i])
      ax.set_xlabel("steps")
      ax.set_title(f"{reduced_by[i]} of locals")


  def _draw_smooth_history(
      self,
      axes: plt.Axes,
      history: List, 
      parameters: Dict,
      power: int = 5
    ):
    if len(history) < power + 1:
      return

    smoothed = make_interp_spline(np.arange(1, len(history) + 1), history, k=power)

    x =  np.linspace(1, len(history), 300)
    axes.plot(x, smoothed(x))
    axes.set_xlabel("steps")
    axes.set_ylabel("function value")
    axes.set_title(f"{parameters}")
  
  def _reduce_client_history(
    self,
    model: Model,
    client_fraction: float,
    reduce_by: List[str] = ["mean"],
  ):   
    # such stupid mechanism allows me to append histories 
    # in the same order as reducers appear in reduce_by
    result = np.zeros((1, model.client_history_manager.histories.shape[1]))
    for reducer in reduce_by: 
      if reducer == "mean":
        # division on client_fraction is for the same reason as for drop out
        result = np.vstack((
          result, 
          model.client_history_manager.mean() / client_fraction
        ))  
      elif reducer == "max":
        result = np.vstack((
          result, 
          model.client_history_manager.max()
        ))

      elif reducer == "min":
        result = np.vstack((
          result, 
          model.client_history_manager.min()
        ))

      elif reducer == "sum":
        result = np.vstack((
          result, 
          model.client_history_manager.sum() / client_fraction
        ))

    return np.delete(result, obj=0, axis=0)
    


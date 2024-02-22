import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from function.api import BaseOptimisationFunction
from optimization.federative.fedavg import BaseFederatedOptimizer
from common.model import Model
from pipeline.history_manager import HistoryManager
from common.distributor import DataDistributor
from common.reducers import reducer

from itertools import product
from functools import reduce
from typing import Dict, List, Tuple, Type
from scipy.interpolate import make_interp_spline
from matplotlib.cm import get_cmap


class Pipeline:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    optimizer: Type,
    metrics: Dict[str, callable],
    parameters: Dict[str, List],
    distributor: DataDistributor, 
    X: np.ndarray,
    y: np.ndarray,
  ):
    self.function: BaseOptimisationFunction = function
    self.optimizer = optimizer
    self.metrics: Dict[callable] = metrics

    self.parameters_keys = parameters.keys()
    self.parameters_lists: List[List] = list(product(*parameters.values()))
    self.parameters_lists_count = len(self.parameters_lists)

    self.distributor: DataDistributor = distributor
    self.X: np.ndarray = X
    self.y: np.ndarray = y

  def run(
      self,
      choose_best_by,
      show_history: bool = False,
      scaled: bool = False,
      reducers: List[reducer] = []
    ) -> Tuple[Model, Dict[str, List]]:
    self.scaled = scaled
    history_managers: Dict[str, HistoryManager] = {}
    best_model: Model = None
    best_metric_value = np.inf
    best_parameters = {}

    fig: plt.Figure = None
    axes: plt.Axes = None
    if show_history:
      num_columns = 2 if reducers is None else len(reducers) + 2
      fig, axes =  plt.subplots(
        len(self.parameters_lists), num_columns, figsize=(7.5 * num_columns, self.parameters_lists_count * 2.5)
      )
      if type(axes) is not np.ndarray or \
         len(self.parameters_lists) == 1 and len(axes) == num_columns:
        axes = np.array([axes]) 
      
    for i, parameters_list in enumerate(self.parameters_lists):
      parameters = dict(zip(self.parameters_keys, parameters_list))
      parameters_key = str(parameters)
      data: Dict[str, Dict] = self.distributor.distribute(
        X=self.X,
        y=self.y,
        n_parts=parameters.pop("n_clients", 16),
        iid_fraction=parameters.pop("iid_fraction", 0.3),
      )

      for data_type in data.keys():
        history_managers.setdefault(data_type, HistoryManager())

      rounds = parameters.pop("rounds", 1)
      model: Model = Model(deepcopy(self.function), data["train"]["X"], data["train"]["y"])
      
      optimizer: BaseFederatedOptimizer = self.optimizer(**parameters)
      for round in range(rounds):
        for data_type, current_data in data.items():
          history_managers[data_type].append(
            parameters_key,
            model.validate(X_val=current_data["X"], y_val=current_data["y"])
          )
          
        optimizer.play_round(model)

          
      if show_history:
        self._draw_history(
          axes=axes,
          history_managers=history_managers,
          reducers=reducers
        )

      X_val, y_val = data["train"]["X"]["server"], data["train"]["y"]["server"]
      if "test" in data.keys():
        X_val, y_val = data["test"]["X"]["server"], data["test"]["y"]["server"]

      print(f"\nFor parameters: {parameters_key}:")
      for key, metric in self.metrics.items():
        computed_metric = metric(y_val, model.server.function.predict(X_val))
        if key == choose_best_by and computed_metric < best_metric_value:
          best_metric_value = computed_metric
          best_parameters = parameters
          best_model = model

        print(f"{key} : {computed_metric}")

    if show_history:
      fig.tight_layout()
      fig.legend(data.keys())

    return best_model, best_parameters


  def _draw_history(
    self,
    axes: np.ndarray[np.ndarray[plt.Axes]],
    history_managers: Dict[str, HistoryManager],
    reducers: List[str] = None
  ):
    colors = [get_cmap("turbo")(value) for value in np.linspace(0.2, 0.9, len(history_managers))]

    for data_type, history_manager in history_managers.items():
      color = colors.pop()
      for i, (key, history) in enumerate(history_manager.history.items()):
        self._draw_all_history(
          title=key,
          color=color,
          horizontal_axes=axes[i],
          server_history=history["server"],
          client_history=history["clients"],
          reducers=reducers
        )


  def _draw_all_history(
    self,
    title: str,
    color,
    horizontal_axes: np.ndarray[plt.Axes], 
    server_history: List, 
    client_history: np.ndarray[np.ndarray],
    reducers: List[reducer]
  ):
    ymin = min(server_history)
    ymax = max(server_history)

    horizontal_axes[0].plot(np.arange(1, len(server_history) + 1), server_history, color=color)
    horizontal_axes[0].set_xlabel("steps")
    horizontal_axes[0].set_ylabel("function value")
    horizontal_axes[0].set_title(f"global with {title}")

    min_mean_max = self._prepare_fill_between(client_history)
    horizontal_axes[1].plot(np.arange(1, len(min_mean_max["mean"]) + 1), min_mean_max["mean"], color=color)
    horizontal_axes[1].fill_between(
      np.arange(1, len(min_mean_max["mean"]) + 1), 
      min_mean_max["min"], 
      min_mean_max["max"], 
      color=color,
      alpha=0.175
    )
    horizontal_axes[1].set_xlabel("steps")
    horizontal_axes[1].set_title(f"min < mean < max of locals")

    ymin = min(ymin, min(min_mean_max["min"]))
    ymax = max(ymax, max(min_mean_max["max"]))
    
    ax: plt.Axes
    for ax, reducer in zip(horizontal_axes[2:], reducers):
      reduced_history = reducer(client_history)

      ax.plot(np.arange(1, len(reduced_history) + 1), reduced_history, color=color)
      ax.set_xlabel("steps")
      ax.set_title(f"{str(reducer)} of locals")

      ymin = min(ymin, min(reduced_history))
      ymax = max(ymax, max(reduced_history))
    
    if self.scaled:
      for ax in horizontal_axes:
        ax.set_ylim(
          (min(ymin, ax.get_ylim()[0]), max(ymax, ax.get_ylim()[1])
        ))
  
  def _prepare_fill_between(
    self,
    history: np.ndarray
  ) -> Dict[str, np.ndarray]:
    results = {}

    results["mean"] = np.nan_to_num(history, nan=0).mean(axis=0)
    results["max"] = np.nan_to_num(history, nan=-np.inf).max(axis=0)
    results["min"] = np.nan_to_num(history, nan=+np.inf).min(axis=0)

    return results
  

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
  
# here `for` loop with
# prepare_new_round after each round
# also it is useful to move new function that will return
# consider plot validation 
"""
dict = {
  str(parameters): {
    "server" : {
      "history" : {
        "val" : ...,
        "train" : ...,
      },

      // useless
      "metrics" : { 
        "mae" : ...,
        ...
      }
    },
    "client" : {
      "history" : {
        "mean" : {
          "val" : ...,
          "train" : ...,
        }
      },
        "max" : {
          "val" : ...,
          "train" : ...,
        }
      },

      // useless
      "metrics" : { 
        "mae" : ..., 
        ...
      }
    }
  }
  subfigures same as numbers of parameters + 1
  +1 is for histories
  and the rest is for metrics groups 
  i.e.
            METRICS GROUPS
  +-------str(parameters1)---------+
  |mae for server | mse for server |
  +---------------+-------=--------+
  |mae for client | mse for client |
  +---------------+----------------+
  |mape for client| mape for client|
  +---------------+----------------+
  +-------str(parameters2)---------+
  |mae for server | mse for server |
  +---------------+-------=--------+
  |mae for client | mse for client |
  +---------------+----------------+
  |mape for client| mape for client|
  +---------------+----------------+
but seems to be useless

consider metrics plot graph by fixed metric

"""
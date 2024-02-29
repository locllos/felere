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
from concurrent.futures import Executor

from tqdm import tqdm

class Pipeline:
  def __init__(
    self,
    function: BaseOptimisationFunction,
    optimizers: List[Type],
    metrics: Dict[str, callable],
    parameters: Dict[str, List],
    distributor: DataDistributor, 
    X: np.ndarray,
    y: np.ndarray,
    executor: Executor = None
  ):
    self.function: BaseOptimisationFunction = function
    self.optimizers: List[Type] = optimizers
    self.metrics: Dict[callable] = metrics

    self.parameters_keys = parameters.keys()
    self.parameters_lists: List[List] = sorted(list(product(*parameters.values())))
    self.parameters_lists_count = len(self.parameters_lists)

    self.generated_data: Dict[float, Dict] = {}
    self.distributor: DataDistributor = distributor
    self.X: np.ndarray = X
    self.y: np.ndarray = y

    self.executor: Executor = executor

  def run(
    self,
    choose_best_by,
    scaled: bool = False,
    reducers: List[reducer] = []
  ) -> Tuple[Model, Dict[str, List]]:
    self.scaled = scaled
    best_model: Model = None
    best_metric_value = np.inf
    best_parameters = {}

    num_columns = 3 + len(self.metrics) if reducers is None else len(reducers) + len(self.metrics) + 3

    main_fig = plt.figure(
      layout='constrained',
      figsize=(7.5 * num_columns * len(self.optimizers), self.parameters_lists_count * 3 * len(self.optimizers))
    )
    subfigs = main_fig.subfigures(len(self.optimizers), 1, wspace=4, hspace=0)
    if type(subfigs) is not np.ndarray:
      subfigs = np.array([subfigs])
      
    for fig_id, optimizer_class in enumerate(self.optimizers):
      history_managers: Dict[str, HistoryManager] = {}

      axes: np.ndarray[plt.Axes] = subfigs[fig_id].subplots(
        len(self.parameters_lists), num_columns
      )
      if type(axes) is not np.ndarray or \
          len(self.parameters_lists) == 1 and len(axes) == num_columns:
        axes = np.array([axes]) 
        
      for i, parameters_list in enumerate(self.parameters_lists):
        parameters = dict(zip(self.parameters_keys, parameters_list))
        parameters_key = str(parameters)

        iid_fraction, n_clients = parameters.pop("iid_fraction", 0.3), parameters.pop("n_clients", 16)
        generation_key = str([iid_fraction, n_clients])
        if generation_key not in self.generated_data.keys():
          self.generated_data[generation_key] = self.distributor.distribute(
            X=self.X,
            y=self.y,
            n_parts=n_clients,
            iid_fraction=iid_fraction,
          )
        data: Dict[str, Dict] = self.generated_data[generation_key]
        for data_type in data.keys():
          history_managers.setdefault(data_type, HistoryManager())

        rounds = parameters.pop("rounds", 1)
        model: Model = Model(
          deepcopy(self.function), 
          data["train"]["X"], data["train"]["y"], parameters.pop("clients_fraction"),
          self.executor
        )

        optimizer: BaseFederatedOptimizer = optimizer_class(**parameters)
        print(f"\n{str(optimizer)} for parameters: {parameters_key}:")
        for round in tqdm(range(rounds), desc="learning"):
          for data_type, current_data in data.items():
            history_managers[data_type].append(
              parameters_key,
              model.validate(X_val=current_data["X"], y_val=current_data["y"], metrics=self.metrics)
            )
            
          optimizer.play_round(model)

            
        self._draw_history(
          axes=axes,
          history_managers=history_managers,
          reducers=reducers
        )

        X_val, y_val = data["train"]["X"]["server"], data["train"]["y"]["server"]
        if "test" in data.keys():
          X_val, y_val = data["test"]["X"]["server"], data["test"]["y"]["server"]

        for key, metric in self.metrics.items():
          computed_metric = metric(model.server.function.predict(X_val), y_val)
          if key == choose_best_by and computed_metric < best_metric_value:
            best_metric_value = computed_metric
            best_parameters = parameters
            best_model = model

          print(f"{key} : {computed_metric}")

        subfigs[fig_id].suptitle(f"{str(optimizer)}", fontsize='x-large')

      subfigs[fig_id].legend(data.keys())
    main_fig.savefig("../res/last.png")
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
          history=history,
          reducers=reducers
        )


  def _draw_all_history(
    self,
    title: str,
    color,
    horizontal_axes: np.ndarray[plt.Axes], 
    history: Dict,
    reducers: List[reducer]
  ):
    ymin = min(history["server"])
    ymax = max(history["server"])

    horizontal_axes[0].plot(np.arange(1, len(history["server"]) + 1), history["server"], color=color)
    horizontal_axes[0].set_xlabel("steps")
    horizontal_axes[0].set_ylabel("function value")
    horizontal_axes[0].set_title(f"global with {title}")

    horizontal_axes[1].semilogy(np.arange(1, len(history["norm_grads"]) + 1), history["norm_grads"], color=color)
    horizontal_axes[1].set_xlabel("steps")
    horizontal_axes[1].set_ylabel("norm value")
    horizontal_axes[1].set_title(f"server gradient norm")
    ymin = min(ymin, min(history["norm_grads"]))
    ymax = max(ymax, max(history["norm_grads"]))

    min_mean_max = self._prepare_fill_between(history["clients"])
    horizontal_axes[2].plot(np.arange(1, len(min_mean_max["mean"]) + 1), min_mean_max["mean"], color=color)
    horizontal_axes[2].fill_between(
      np.arange(1, len(min_mean_max["mean"]) + 1), 
      min_mean_max["min"], 
      min_mean_max["max"], 
      color=color,
      alpha=0.175
    )
    horizontal_axes[2].set_xlabel("steps")
    horizontal_axes[2].set_title(f"min < mean < max of locals")
    ymin = min(ymin, min(min_mean_max["min"]))
    ymax = max(ymax, max(min_mean_max["max"]))

    
    ax: plt.Axes
    for ax, reducer in zip(horizontal_axes[3:3+len(reducers)], reducers):
      reduced_history = reducer(history["clients"])

      ax.plot(np.arange(1, len(reduced_history) + 1), reduced_history, color=color)
      ax.set_xlabel("steps")
      ax.set_title(f"{str(reducer)} of locals")

      ymin = min(ymin, min(reduced_history))
      ymax = max(ymax, max(reduced_history))
    
    ax: plt.Axes
    for ax, (metric, results) in zip(horizontal_axes[3 + len(reducers):], history["metrics"].items()):
      reduced_history = reducer(history["clients"])

      ax.plot(np.arange(1, len(results) + 1), results, color=color)
      ax.set_xlabel("steps")
      ax.set_title(f"{metric}")

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
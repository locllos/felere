import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from felere.function.api import BaseOptimisationFunction
from felere.optimization.federative.fedavg import BaseFederatedOptimizer
from felere.common.simulation import Simulation
from felere.pipelines.history_manager import HistoryManager
from felere.common.distributor import DataDistributor
from felere.common.reducers import reducer

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
    metrics: Dict[str, callable],
    optimizer_parameters: Dict[str, Dict[str, List]],
    distributor: DataDistributor, 
    X: np.ndarray,
    y: np.ndarray,
    subplot_width: int = 3,
    subplot_height: int = 5,
    font_size: int = 16
  ):
    self.title_size: int = font_size
    self.function: BaseOptimisationFunction = function
    self.metrics: Dict[callable] = metrics

    self.parameters_lists_count = 0
    self.optimizers_parameters_combinations: Dict[str, List[List]] = {}
    for optimizer, parameters in optimizer_parameters.items():
      self.optimizers_parameters_combinations[optimizer] = {
        "combinations" : sorted(list(product(*parameters.values()))),
        "keys"  : parameters.keys()
      }
      self.parameters_lists_count += \
        len(self.optimizers_parameters_combinations[optimizer]["combinations"])

    self.generated_data: Dict[float, Dict] = {}
    self.distributor: DataDistributor = distributor
    self.X: np.ndarray = X
    self.y: np.ndarray = y
    self.subplot_width: int = subplot_width
    self.subplot_height: int = subplot_height

  def run(
    self,
    choose_best_by,
    scaled: bool = False,
    with_grads: bool = False,
    reducers: List[reducer] = [],
    plot_name: str = "last.png"
  ) -> Tuple[Simulation, Dict[str, List]]:
    self.scaled = scaled
    self.with_grads = with_grads
    best_model: Simulation = None
    best_metric_value = np.inf
    best_parameters = {}

    num_columns = 2 + len(self.metrics) if reducers is None else len(reducers) + len(self.metrics) + 2
    if self.with_grads:
      num_columns += 1

    subplot_width = 2 * self.subplot_width / len(self.optimizers_parameters_combinations) if len(self.optimizers_parameters_combinations) > 1 else self.subplot_width * 2
    subplot_height = self.subplot_height * 0.5
    main_fig = plt.figure(
      layout='constrained',
      figsize=(subplot_width * num_columns * len(self.optimizers_parameters_combinations), self.parameters_lists_count * subplot_height)
    )
    subfigs = main_fig.subfigures(len(self.optimizers_parameters_combinations), 1)
    if type(subfigs) is not np.ndarray:
      subfigs = np.array([subfigs])
      
    for optimizer_id, (optimizer_type, parameters) in enumerate(self.optimizers_parameters_combinations.items()):
      history_managers: Dict[str, HistoryManager] = {}

      axes: np.ndarray[plt.Axes] = subfigs[optimizer_id].subplots(
        len(parameters["combinations"]), num_columns
      )
      if type(axes) is not np.ndarray or \
          len(parameters["combinations"]) == 1 and len(axes) == num_columns:
        axes = np.array([axes]) 
        
      for i, parameters_list in enumerate(parameters["combinations"]):
        current_parameters = dict(zip(parameters["keys"], parameters_list))
        parameters_key = self._pretty_parameters_string(current_parameters)

        iid_fraction, n_clients = current_parameters.pop("iid_fraction", 0.3), current_parameters.pop("n_clients", 16)
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

        rounds = current_parameters.pop("rounds", 1)
        model: Simulation = Simulation(
          deepcopy(self.function), 
          data["train"]["X"], data["train"]["y"], current_parameters.pop("clients_fraction"),
        )

        optimizer: BaseFederatedOptimizer = optimizer_type(**current_parameters)
        print(f"\n{optimizer} for parameters: {parameters_key}:")
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
            best_parameters = current_parameters
            best_model = model

          print(f"{key} : {computed_metric}")

        subfigs[optimizer_id].suptitle(f"{optimizer}", fontsize=27)

      subfigs[optimizer_id].legend(data.keys(), fontsize=self.title_size)
    main_fig.savefig(plot_name)
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

    current_plot = 0
    horizontal_axes[current_plot].plot(np.arange(1, len(history["server"]) + 1), history["server"], color=color)
    horizontal_axes[current_plot].set_xlabel("rounds", fontsize=self.title_size)
    horizontal_axes[current_plot].set_ylabel("loss", fontsize=self.title_size)
    horizontal_axes[current_plot].set_title(f"global with {title}", fontsize=self.title_size)
    current_plot += 1

    if self.with_grads:
      horizontal_axes[current_plot].semilogy(np.arange(1, len(history["norm_grads"]) + 1), history["norm_grads"], color=color)
      horizontal_axes[current_plot].set_xlabel("rounds", fontsize=self.title_size)
      horizontal_axes[current_plot].set_ylabel("norm value", fontsize=self.title_size)
      horizontal_axes[current_plot].set_title(f"server gradient norm", fontsize=self.title_size)
      ymin = min(ymin, min(history["norm_grads"]))
      ymax = max(ymax, max(history["norm_grads"]))

      current_plot += 1

    min_mean_max = self._prepare_fill_between(history["clients"])
    horizontal_axes[current_plot].plot(np.arange(1, len(min_mean_max["mean"]) + 1), min_mean_max["mean"], color=color)
    horizontal_axes[current_plot].fill_between(
      np.arange(1, len(min_mean_max["mean"]) + 1), 
      min_mean_max["min"], 
      min_mean_max["max"], 
      color=color,
      alpha=0.175
    )
    horizontal_axes[current_plot].set_xlabel("rounds", fontsize=self.title_size)
    horizontal_axes[current_plot].set_title(f"min < mean < max of locals", fontsize=self.title_size)
    ymin = min(ymin, min(min_mean_max["min"]))
    ymax = max(ymax, max(min_mean_max["max"]))
    current_plot += 1

    
    ax: plt.Axes
    for ax, reducer in zip(horizontal_axes[current_plot:current_plot+len(reducers)], reducers):
      reduced_history = reducer(history["clients"])

      ax.plot(np.arange(1, len(reduced_history) + 1), reduced_history, color=color)
      ax.set_xlabel("rounds", fontsize=self.title_size)
      ax.set_title(f"{str(reducer)} of locals", fontsize=self.title_size)

      ymin = min(ymin, min(reduced_history))
      ymax = max(ymax, max(reduced_history))
    
    ax: plt.Axes
    for ax, (metric, results) in zip(horizontal_axes[current_plot + len(reducers):], history["metrics"].items()):
      ax.plot(np.arange(1, len(results) + 1), results, color=color)
      ax.set_xlabel("rounds", fontsize=self.title_size)
      ax.set_title(f"{metric}: best={round(max(results), 3)}", fontsize=self.title_size)
    
    if self.scaled:
      for ax in horizontal_axes[:-len(history["metrics"])]:
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
    axes.set_xlabel("rounds", fontsize=self.title_size)
    axes.set_ylabel("function value", fontsize=self.title_size)
    axes.set_title(f"{parameters}", fontsize=self.title_size)

  def _pretty_parameters_string(self, parameters: Dict[str, List[float]], line_break_at_every=4):
    beautified: str = ""
    for i, (key, param) in enumerate(parameters.items()):
      if i > 0 and i % line_break_at_every == 0 and i - 1 < len(parameters):
        beautified = beautified + '\n'
      elif i > 0:
        beautified += ", "
      
      if type(param) == float:
        beautified = f"{beautified} {key}={round(param, 8)}"
      else:
        beautified = f"{beautified} {key}={param}"



    return beautified
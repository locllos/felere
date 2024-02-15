import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from .function import BaseOptimisationFunction
from optimization.federative.fedavg import BaseFederatedOptimizer
from common.model import Model



from itertools import product
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
      show_global_history: bool | str = False # show_history \in [False, True, "smooth"]
    ) -> Tuple[Model, Dict[str, List]]:
    best_model: BaseOptimisationFunction = None
    best_metric_value = np.inf
    best_parameters = {}

    fig: plt.Figure = None
    axes: plt.Axes = None
    if show_global_history:
      fig, axes =  plt.subplots(
        len(self.parameters_lists), 1, figsize=(12, self.parameters_lists_count * 2.5)
      )
      axes = axes if type(axes) is np.ndarray else [axes]
      
    for i, parameters_list in enumerate(self.parameters_lists):
      parameters = dict(zip(self.parameters_keys, parameters_list))
      rounds = parameters.pop("rounds", 1)

      optimizer: BaseFederatedOptimizer = self.optimizer(**parameters)
      model = deepcopy(self.model)
      
      optimizer.optimize(model, rounds)
      if show_global_history:
        if type(show_global_history) is str and show_global_history == "smooth":
          self._draw_smooth_history(axes[i], model.server.history, parameters)
        else:
          self._draw_history(axes[i], model.server.history, parameters)
        
      print(f"\nFor parameters: {parameters}:")
      for key, metric in self.metrics.items():
        computed_metric = metric(self.y_val, model.server.function.predict(self.X_val))
        if key == choose_best_by and computed_metric < best_metric_value:
          best_metric_value = computed_metric
          best_parameters = parameters
          best_model = model

        print(f"{key} : {computed_metric}")
    
    if show_global_history:
      fig.tight_layout()

    return best_model, best_parameters


  def _draw_history(self, axes: plt.Axes, history: List, parameters: Dict):
    axes.plot(np.arange(1, len(history) + 1), history)
    axes.set_xlabel("steps")
    axes.set_ylabel("function value")
    axes.set_title(f"{parameters}")


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
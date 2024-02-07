import numpy as np

from function import BaseOptimisationFunction
from optimization.federative import BaseFederatedOptimizer

import matplotlib.pyplot as plt

from itertools import product
from typing import Dict, List

class Pipeline:
  def __init__(
    self,
    optimizer: BaseFederatedOptimizer,
    metrics: Dict[str, callable],
    parameters: Dict[str, List], 
    X_test: np.ndarray, 
    y_test: np.ndarray
  ):
    self.optimizer: BaseFederatedOptimizer = optimizer
    self.metrics: Dict[callable] = metrics

    self.parameters_keys = parameters.keys()
    self.parameters_lists: List[List] = list(product(*parameters.values()))
    self.parameters_lists_count = len(self.parameters_lists)

    self.X_test = X_test
    self.y_test = y_test
  
  def run(self, show_history=False, choose_best_by=None) -> BaseOptimisationFunction:
    best_function: BaseOptimisationFunction = None
    best_metric_value = np.inf

    fig: plt.Figure = None
    axes: plt.Axes = None
    if show_history:
      fig, axes =  plt.subplots(
        len(self.parameters_lists), 1, figsize=(12, self.parameters_lists_count * 2.5)
      )
      axes = axes if type(axes) is np.ndarray else [axes]
      
    for i, parameters_list in enumerate(self.parameters_lists):
      history = []
      parameters = dict(zip(self.parameters_keys, parameters_list))

      if show_history:
        function, history = self.optimizer.optimize(return_history=True, **parameters)
        self._draw_history(axes[i], history, parameters)
      else:
        function = self.optimizer.optimize(**parameters)
        
      print(f"\nFor parameters: {parameters}:")
      for key, metric in self.metrics.items():
        computed_metric = metric(self.y_test, function.predict(self.X_test))
        if key == choose_best_by and computed_metric < best_metric_value:
          best_metric_value = computed_metric
          best_function = function

        print(f"{key} : {computed_metric}")
    
    if show_history:
      fig.tight_layout()

    return best_function

  def _draw_history(self, axes: plt.Axes, history: List, parameters: Dict):
    axes.plot(np.arange(1, len(history) + 1), history)
    axes.set_xlabel("steps")
    axes.set_ylabel("function value")
    axes.set_title(f"{parameters}")
import numpy as np

from common.function import BaseOptimisationFunction

def simple_gradient_descent(
    function: BaseOptimisationFunction,
    X: np.ndarray, 
    y: np.ndarray,
    eta: float = 1e-4,
    steps: int = 128,
    return_history = False
  ) -> list | None:
  w = None
  history = []
  for _ in range(steps):
    history.append(function(X=X, y=y))
    
    step = (-1) * eta * function.grad(w)
    w = function.update(step)
    
  if return_history:
    return history
import torch
import numpy as np

from function.api import BaseOptimisationFunction

def simple_gradient_descent(
    function: BaseOptimisationFunction,
    X: torch.Tensor, 
    y: torch.Tensor,
    eta: float = 1e-4,
    steps: int = 128,
    return_history = False
  ) -> list | None:
  history = []
  for _ in range(steps):
    history.append(function(X=X, y=y))
    
    step = (-1) * eta * function.grad()
    function.update(step)
    
  if return_history:
    return history
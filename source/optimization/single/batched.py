import numpy as np

from function.api import BaseOptimisationFunction

def batched_gradient_descent(
    function: BaseOptimisationFunction, 
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    eta: float = 1e-3,
    epochs: int = 32,
    return_history = False
) -> list | None:
  def batch_generator(X: np.ndarray, y: np.ndarray):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
      yield X[i: min(i + batch_size, n_samples)], y[i: min(i + batch_size, n_samples)]
  
  history = []
  for _ in range(epochs):
    for X_batch, y_batch in batch_generator(X, y):
      history.append(function(X=X_batch, y=y_batch))
      
      step = (-1) * eta * function.grad() / batch_size
      function.update(step)
    
  if return_history:
    return history

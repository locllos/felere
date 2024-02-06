import numpy as np
import matplotlib.pyplot as plt

from common.function import BaseOptimisationFunction

def draw_history(history):
  plt.clf()
  plt.plot(np.arange(1, len(history) + 1), history)
  plt.xlabel("steps")
  plt.ylabel("function value")

def simple_gradient_descent(
    function: BaseOptimisationFunction,
    X: np.ndarray, 
    y: np.ndarray,
    eta: float = 1e-4,
    steps: int = 128,
    show_history = False
  ):
  w = None
  history = []
  for _ in range(steps):
    history.append(function(X=X, y=y))
    
    step = (-1) * eta * function.grad(w)
    w = function.update(step)
    
  if show_history:
    draw_history(history=history)


def batched_gradient_descent(
    function: BaseOptimisationFunction, 
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    eta: float = 1e-3,
    epochs: int = 32,
    show_history = False
):
  def batch_generator(X: np.ndarray, y: np.ndarray):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
      yield X[i: min(i + batch_size, n_samples)], y[i: min(i + batch_size, n_samples)]
  
  w = None
  history = []
  for _ in range(epochs):
    for X_batch, y_batch in batch_generator(X, y):
      history.append(function(X=X_batch, y=y_batch))
      
      step = (-1) * eta * function.grad(w) / batch_size
      w = function.update(step)
    
  if show_history:
    draw_history(history=history)
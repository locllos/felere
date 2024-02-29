from typing import List
from .api import BaseOptimisationFunction, np

import torch
from torch import nn

class TorchFunction(BaseOptimisationFunction):
  def __init__(self, module: nn.Module, loss_fn):
    self.module: nn.Module = module
    self.loss_fn = loss_fn
    self.loss: torch.Tensor = torch.tensor([])
    self.flattener: TorchFunction.Flattener = TorchFunction.Flattener(module.parameters())
    self.last_gradient: np.ndarray = None

  def __call__(self, X: np.ndarray, y: np.ndarray, requires_grad=True):
    if requires_grad:
      return self._compute_function(X, y).clone().detach().numpy(force=True)

    with torch.no_grad():
      return self._compute_function(X, y).clone().detach().numpy(force=True)
  
  def grad(self):
    self.loss.backward()
    
    for parameters in self.module.parameters():
      if parameters.grad is None:
        print("parameters.grad is none")
        self.loss.backward()

    grads: List[torch.Tensor] = []
    for parameters in self.module.parameters():
      grads.append(parameters.grad.clone().detach())

    self.module.zero_grad()
    self.last_gradient = self.flattener.flatten(grads)
    return self.last_gradient

  def last_grad(self) -> np.ndarray:
    return self.last_gradient

  def update(self, step: np.ndarray):
    step = self.flattener.unflatten(step)

    with torch.no_grad():
      for update, parameters in zip(step, self.module.parameters()):
        parameters += update
  
  def predict(self, X: np.ndarray):
    with torch.no_grad():
      return self.module.forward(torch.tensor(X)).clone().detach().numpy(force=True)
      

  def weights(self) -> np.ndarray:
    parameters_list: List[torch.Tensor] = []
    for parameters in self.module.parameters():
      parameters_list.append(parameters.clone().detach())

    return self.flattener.flatten(parameters_list)
  
  
  class Flattener:
    def __init__(self, parameters_generator):
      self.shapes: List[torch.Size] = []

      parameters: nn.Parameter
      for parameters in parameters_generator:
        self.shapes.append(parameters.shape)

    def flatten(self, arrays: List[torch.Tensor]) -> np.ndarray:
      flat = np.array([])
      for array in arrays:
        flat = np.append(flat, array.numpy(force=True).flatten())

      return flat

    
    def unflatten(self, flat: np.ndarray) -> List[torch.Tensor]:
      arrays: List[torch.Tensor] = []
      
      start = 0
      for shape in self.shapes:
        arrays.append(torch.from_numpy(flat[start : start + shape.numel()].reshape(shape)))
        
        start += shape.numel()

      return arrays
    
  def _compute_function(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
    # write type convertor float -> Float, int -> Long
    if self.loss_fn is None:
      self.loss = self.module.forward(torch.tensor(X))
    else:
      self.loss = self.loss_fn(self.module.forward(torch.tensor(X)), torch.tensor(y))
    
    return self.loss
    
 
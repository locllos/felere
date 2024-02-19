from typing import List
from .api import BaseOptimisationFunction

import torch
from torch import nn


class TorchFunction(BaseOptimisationFunction):
  def __init__(self, model: nn.Module, loss_fn = None):
    self.model: nn.Module = model
    self.loss_fn = loss_fn
    self.loss: torch.Tensor = torch.Tensor()
    self.flattener: TorchFunction.Flattener = TorchFunction.Flattener(model.parameters())


  def __call__(self, X: torch.Tensor , y: torch.Tensor):
    if self.loss_fn is None:
      self.loss: torch.Tensor = self.model.forward(torch.Tensor(X))
    else:
      self.loss: torch.Tensor = self.loss_fn(self.model.forward(torch.Tensor(X)), torch.Tensor(y))
      
    
    return self.loss.clone().detach()
  
  def grad(self):
    self.loss.backward()
    
    grads: List[torch.Tensor] = []
    for parameters in self.model.parameters():
      grads.append(parameters.grad.clone().detach())

    self.model.zero_grad()
    return self.flattener.flatten(grads)

  def update(self, step: torch.Tensor):
    step = self.flattener.unflatten(step)

    with torch.no_grad():
      for update, parameters in zip(step, self.model.parameters()):
        parameters += update
  
  def predict(self, X: torch.Tensor):
    return self.model.forward(torch.Tensor(X))

  def weights(self) -> torch.Tensor:
    parameters_list: List[torch.Tensor] = []
    for parameters in self.model.parameters():
      parameters_list.append(parameters.clone().detach())

    return self.flattener.flatten(parameters_list)
  
  
  class Flattener:
    def __init__(self, parameters_generator):
      self.shapes: List[torch.Size] = []

      parameters: nn.Parameter
      for parameters in parameters_generator:
        self.shapes.append(parameters.shape)

    def flatten(self, arrays: List[torch.Tensor]) -> torch.Tensor:
      flat = torch.Tensor([])
      for array in arrays:
        flat = torch.cat((flat, array))

      return flat

    
    def unflatten(self, flat: torch.Tensor) -> List[torch.Tensor]:
      arrays: List[torch.Tensor] = []
      
      start = 0
      for shape in self.shapes:
        arrays.append(flat[start : start + shape.numel()].reshape(shape))
        
        start += shape.numel()

      return arrays
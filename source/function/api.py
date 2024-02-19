import torch

class BaseOptimisationFunction:
  def __call__(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs):
    raise NotImplementedError
  
  def grad(self, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError
  
  def update(self, step, *args, **kwargs):
    raise NotImplementedError
  
  def predict(self, X: torch.Tensor = None, *args, **kwargs):
    raise NotImplementedError

  def weights(self) -> torch.Tensor:
    raise NotImplementedError
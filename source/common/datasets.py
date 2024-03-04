from typing import Tuple
from sklearn.datasets import load_digits, make_regression
from torchvision.datasets import CIFAR10

import numpy as np

import os

class BaseDataset():
  def generate(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
  

class LinearDataset(BaseDataset):
  @staticmethod
  def generate(
      n_samples, 
      n_features,
      n_targets,
      bias,
      noise,
      coef=True,
      random_state=42
    ) -> Tuple[int, np.ndarray, int, np.ndarray]:
    return make_regression(
      n_samples=n_samples,
      n_features=n_features,
      n_informative=n_features, 
      n_targets=n_targets,
      coef=coef,
      bias=bias,
      noise=noise,
      random_state=random_state
    )
  

class MNISTDataset(BaseDataset):
  @staticmethod
  def generate(n_classes=10) -> Tuple[np.ndarray, np.ndarray]:
    return load_digits(n_class=n_classes, return_X_y=True)



class CIFAR10Dataset(BaseDataset):
  @staticmethod
  def generate(to_float=False) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists("../res/"):
      os.mkdir("../res/")
    
    train_dataset = CIFAR10("./../res/cifar.data", download=True, train=False)
    test_dataset = CIFAR10("./../res/cifar.data", download=True, train=True)


    X = np.vstack((train_dataset.data, test_dataset.data)).transpose((0, 3, 1, 2))
    y = np.hstack((train_dataset.targets, test_dataset.targets))

    if to_float:
      X = np.float32(X)
      X /= 256
    
    return X, y

    

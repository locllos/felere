from typing import Tuple
from sklearn.datasets import load_digits, make_regression
from torchvision.datasets import CIFAR10, FashionMNIST

import numpy as np

import re
import os
import requests
import pymorphy2
from functools import reduce
from nltk.tokenize import word_tokenize


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

    
class FashionMNISTDataset(BaseDataset):
  @staticmethod
  def generate(to_float=False) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists("../res/"):
      os.mkdir("../res/")
    
    train_dataset = FashionMNIST("./../res/fmnist.data", download=True, train=True)
    test_dataset = FashionMNIST("./../res/fmnist.data", download=True, train=False)

    X = np.vstack((train_dataset.data, test_dataset.data))
    y = np.hstack((train_dataset.targets, test_dataset.targets))

    if to_float:
      X = np.float32(X)
      X /= 256
    
    return X, y

class SherlockInputIdsDataset(BaseDataset):
  @staticmethod
  def generate(context_length=8) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists("../res/"):
      os.mkdir("../res/")
      data = requests.get("https://www.gutenberg.org/files/1661/1661-0.txt")
    if not os.path.exists("../res/sherlock.data/sherlock.txt"):
      os.mkdir("../res/sherlock.data")
      file = open("../res/sherlock.data/sherlock.txt", "w")
      file.write(data.content.decode("utf-8"))
      file.close()
    
    text = reduce(
      SherlockInputIdsDataset._combiner,
      open("../res/sherlock.data/sherlock.txt", "r").read().split("\n")
    )
    text = text[text.find(SherlockInputIdsDataset.kBeginning) + 
                len(SherlockInputIdsDataset.kBeginning) : text.find(SherlockInputIdsDataset.kEnding)]
    
    words = SherlockInputIdsDataset._preprocess_text(text)
    word2id = {
      word : id
      for id, word in enumerate(np.unique(words))
    }
    ids = [word2id[word] for word in words]

    X: np.ndarray = np.array([]) 
    y: np.ndarray = np.array([])
    for pos, input_id in enumerate(ids[context_length:], start=context_length):
      if X.size == 0:
        X = np.array(ids[pos - context_length : pos])
      else:
        X = np.vstack((X, ids[pos - context_length : pos]))
        
      y = np.append(y, input_id)

    return X, y 


  @staticmethod
  def _combiner(combined, current):
    if len(current) == 0:
      return combined
    if combined[-1] == ' ':
      return combined + current
    return combined + ' ' + current

  kBeginning = "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
  kEnding = "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"

  @staticmethod
  def _preprocess_text(text: str):
    morph = pymorphy2.MorphAnalyzer()
    text = text.lower()
    text = re.sub(r'[^ -~]', '', text) # allow ascii only

    words = word_tokenize(text, language='english')
    processed_words = []
    for word in words:
        words = morph.parse(word)[0].normal_form
        processed_words.append(word)

    return processed_words

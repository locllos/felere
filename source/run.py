import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1)

from sklearn.metrics import (
  f1_score
)

if not os.path.exists("./results/"):
  os.mkdir("./results/")

############################
old_print = print
out = open('./results/output.txt', 'w')
from contextlib import redirect_stdout

print = lambda *args, **kw: old_print(*args, **kw) or old_print(*args, file=out, **kw)
###########################
from common.datasets import FashionMNISTDataset
X, y = FashionMNISTDataset().generate(to_float=True)

n_classes = len(np.unique(y))
print(f"{n_classes=}")

print(f"{X.shape=}, {y.shape=}")

n_features = X.shape[1] * X.shape[2]
n_targets = n_classes

print("reshaping X...")
X = np.float32(X).reshape((X.shape[0], X.shape[1] * X.shape[2]))
print(f"{X.dtype=}")
print(f"{X.shape=}")


from common.distributor import DataDistributor
from optimization.federative.fedavg import FederatedAveraging
from optimization.federative.fedprox import FedProx
from optimization.federative.scaffnew import Scaffnew
from optimization.federative.scaffold import Scaffold
from optimization.federative.fedfair import FedFair

from function.torch import TorchFunction

import torch

from common.torch_models import FashionMNISTLinearModel


torch_model = FashionMNISTLinearModel(n_features=n_features, n_targets=n_classes)
function = TorchFunction(torch_model, torch.nn.CrossEntropyLoss())


print(f"{torch_model.n_paramaters()=}")


distributor = DataDistributor(test_size=0.2, server_fraction=0.2)

    
from pipeline.pipeline import Pipeline


pipeline_objective = "./results/n_clients_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [8, 32, 128],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedProx : {
    "n_clients" : [8, 32, 128],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "mu": [0.5], # , 1e-2
  },
  Scaffold : {
    "n_clients" : [8, 32, 128],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedFair : {
    "n_clients" : [8, 32, 128],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "lmbd" : [1e-2]
  },
  Scaffnew : {
    "n_clients" : [8, 32, 128],
    "iid_fraction" : [0.2],
    "clients_fraction": [1],
    "batch_size": [256], 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "proba" : [1/96]
  }
}

optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [8],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [3],
    "eta": [0.5e-2], # , 1e-2
  },
  Scaffold : {
    "n_clients" : [8],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [3],
    "eta": [0.5e-2], # , 1e-2
  },
}


metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

print("done")

#########################################################
pipeline_objective = "./results/iid_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [32],
    "iid_fraction" : [0, 0.5, 1],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedProx : {
    "n_clients" : [32],
    "iid_fraction" : [0, 0.5, 1],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "mu": [0.5], # , 1e-2
  },
  Scaffold : {
    "n_clients" : [32],
    "iid_fraction" : [0, 0.5, 1],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedFair : {
    "n_clients" : [32],
    "iid_fraction" : [0, 0.5, 1],
    "clients_fraction": [0.4],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "lmbd" : [1e-2]
  },
  Scaffnew : {
    "n_clients" : [32],
    "iid_fraction" : [0, 0.5, 1],
    "clients_fraction": [1],
    "batch_size": [256], 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "proba" : [1/96]
  }
}

metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

best, best_params = pipeline.run(
  choose_best_by="f1_score",
  scaled=False,
  with_grads=True,
  reducers=[],
  plot_name=pipeline_objective
)

print("done")

#########################################################
pipeline_objective = "./results/clients_fraction_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.1, 0.5, 1],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedProx : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.1, 0.5, 1],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "mu": [0.5], # , 1e-2
  },
  Scaffold : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.1, 0.5, 1],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  Scaffnew : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.1, 0.5, 1],
    "batch_size": [256], 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "proba" : [1/96]
  }
}

metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

print("done")

#########################################################
pipeline_objective = "./results/epochs_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [16, 64, 256], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedProx : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [16, 64, 256], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "mu": [0.5], # , 1e-2
  },
  Scaffold : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [16, 64, 256], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
  },
  FedFair : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [16, 64, 256], # 16, 64, 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "lmbd" : [1e-2]
  },
  Scaffnew : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "rounds": [64],
    "eta": [0.5e-2], # , 1e-2
    "proba" : [1/16, 1/64, 1/256]
  }
}

metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

print("done")

#########################################################
pipeline_objective = "./results/fedfair_lmbd_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FedFair : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [12, 48, 128],
    "eta": [0.5e-2], # , 1e-2
    "lmbd" : [0.1e-2, 1e-2, 10e-2],
  }
}

metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

print("done")

#########################################################
pipeline_objective = "./results/fedprox_mu_dependency"

print(f"{pipeline_objective=}")
optimizer_parameters = {
  FedProx : {
    "n_clients" : [32],
    "iid_fraction" : [0.2],
    "clients_fraction": [0.3],
    "batch_size": [256], 
    "epochs": [96], # 16, 64, 
    "rounds": [12, 48, 128],
    "eta": [0.5e-2], # , 1e-2
    "mu": [0.1, 0.5, 2], # , 1e-2
  },
}

metrics = {
  "f1_score" : lambda y_proba, y_true: f1_score(np.argmax(y_proba, axis=1), y_true, average="weighted")
}

pipeline = Pipeline(
  function=function,
  metrics=metrics,
  optimizer_parameters=optimizer_parameters,
  distributor=distributor,
  X=X,
  y=y
)

with redirect_stdout(out):
  best, best_params = pipeline.run(
    choose_best_by="f1_score",
    scaled=False,
    with_grads=True,
    reducers=[],
    plot_name=pipeline_objective
  )

print("done")

print("Success: all pipelines were done!")

############################
out.close()
###########################
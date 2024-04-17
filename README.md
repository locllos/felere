# FeLeRe

Federated learning library for researchers.

## About

This library provides an easy-to-use  API and infrastructure for performing analysis and research in the field of federated learning. 

It relies on the simulation of client-server communications, which takes into account client latency and heterogeneous data distributions across all clients.

## More About FeLeRe

Usually, FeLeRe can be used in either of the following ways:

* Testing optimization methods for robustness in typical federated learning settings (client latency, heterogenous data) for proof-of-concept purposes
* Comparing various methods against each other

### Components

At a granular level, FeLeRe is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **felere** | Federated learning library for research with simulated client-server communications. |
| **felere.pipeline** | Pipelines for algorithm testing and analysis |
|**felere.function**| Customizable or pre-implemented functions to be optimized |
|**felere.optimization**| Customizable or pre-implemented optimization methods |
| **felere.common.distributor** | Data ditribution for federated learning clients, considering heterogeneity |
| **felere.common.simulation** | Client-server communication simulation, client sampling and model updating|
| **felere.common.datasets** | Easy-to-load pre-defined datasets |

### Unique feature

Control the heterogeneous and homogeneous distribution of data between all clients to simulate real-life conditions. Depending on the iid_fraction parameter, your data may be distributed.

Depending on the `iid_fraction` parameter, your data may be distributed as:

![distr-example](./res/readme/distr_example.png)

### Implemented methods

1. [FederatedAveraging](https://arxiv.org/abs/1602.05629)
2. [FedProx](https://arxiv.org/abs/1812.06127)
3. [Scaffold](https://arxiv.org/abs/1910.06378)
4. [Scaffnew](https://arxiv.org/abs/2202.09357)
5. [FedFair](https://arxiv.org/abs/2402.16028)

## Usage

### Methods comparison

In order to compare the methods, you can define the Python dictionary in a JSON-like format. Specifically, if we want to compare the `FederatedAveraging` and `Scaffold` methods in the context of full heterogeneity, we need to create a dictionary.:

```python
optimizer_parameters = {
  FederatedAveraging : {
    "n_clients" : [96],
    "iid_fraction" : [0.0], 
    "clients_fraction": [0.2],
    "batch_size": [256], 
    "epochs": [128],  
    "rounds": [16],
    "eta": [0.5e-2],
  },
  Scaffold : {
    "n_clients" : [96],
    "iid_fraction" : [0.1],
    "clients_fraction": [0.2],
    "batch_size": [256], 
    "epochs": [128],
    "rounds": [16],
    "eta": [0.5e-2],a
  }
}
```

And pass it to `felere.pipeline.Pipeline` class constructor, and then run it. This will provide you the output:

![comparision](./res/readme/comparision.png)

From which we can deduce that the `Scaffold` is more stable than `FederagedAveraging`.

### Method implementation

In order to implement a new federated learning method, we need to be inherited from `BaseFederatedOptimizer`, and then implement `play_round` and `client_update` methods, i.e.
we think of the new *awesome* federated learning algorithm, and implement methods `play_round` and `client_update`:

```python
class Custom(BaseFederatedOptimizer):
  def __init__(self, eta):
    self.eta: float = eta      
    
  def play_round(self, model: Simulation):
    _, clients_weights, other = model.clients_update(self.client_update)
    clients_n_samples = other["n_samples"]
      
    next_global_weights = \
      (clients_weights * clients_n_samples).sum(axis=0) / clients_n_samples.sum()
    
    model.server.function.update(
      (-1) * (model.server.function.weights() - next_global_weights)
    )

  def client_update(self, server, client):
    client.function.update(
      (-1) * (client.function.weights() - server.function.weights())
    )
    client.function(X=client.X, y=client.y)

    step = (-1) * self.eta * client.function.grad()
    client.function.update(step)
    
    client.other["n_samples"] = client.X.shape[0]
    return client
  

  def __repr__(self):
    return "CustomMethod"
```

Then we run it on pipeline

## Example of usage

![readme-pipeline](./res/readme/readme-pipeline.gif)

## References

1. Communication-Efficient Learning of Deep Networks from Decentralized Data - https://arxiv.org/abs/1602.05629

2. Federated Optimization in Heterogeneous Networks - https://arxiv.org/abs/1812.06127

3. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning - https://arxiv.org/abs/1910.06378

4. ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally! - https://arxiv.org/abs/2202.09357

5. FedFDP: Federated Learning with Fairness and Differential Privacy - https://arxiv.org/abs/2402.16028
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
| **felere.pipeline** | Algorithm testing and analysis management |
|**felere.function**| Customizable or pre-implemented functions to be optimized |
|**felere.optimization**| Customizable or pre-implemented optimization methods |
| **felere.common.distributor** | Data ditribution for federated learning clients, considering heterogeneity |
| **felere.common.simulation** | Client-server communication simulation, client sampling and model updating|
| **felere.common.datasets** | Easy-to-load pre-defined datasets for convenience |

### Unique feature

Control the heterogeneous and homogeneous of data distribution between all clients in order to simulate real-life conditions.
Depending on the `iid_fraction` parameter, your data may be distributed as:

![distr-example](./res/readme/distr_example.png)

### Implemented methods

1. [FederatedAveraging](https://arxiv.org/abs/1602.05629)
2. [FedProx](https://arxiv.org/abs/1812.06127)
3. [Scaffold](https://arxiv.org/abs/1910.06378)
4. [Scaffnew](https://arxiv.org/abs/2202.09357)
5. [FedFair](https://arxiv.org/abs/2402.16028)

## Usage

### Methods comparing

Adjustable 

## Example of usage

![readme-pipeline](./res/readme/readme-pipeline.gif)

## References

1. Communication-Efficient Learning of Deep Networks from Decentralized Data - https://arxiv.org/abs/1602.05629

2. Federated Optimization in Heterogeneous Networks - https://arxiv.org/abs/1812.06127

3. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning - https://arxiv.org/abs/1910.06378

4. ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally! - https://arxiv.org/abs/2202.09357

5. FedFDP: Federated Learning with Fairness and Differential Privacy - https://arxiv.org/abs/2402.16028
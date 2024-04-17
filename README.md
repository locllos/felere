# FeLeRe

Federated learning library for researchers.

## About

This library provides an easy-to-use  API and infrastructure for performing analysis and research in the field of federated learning. 

It relies on the simulation of client-server communications, which takes into account client latency and heterogeneous data distributions across all clients.

## More About FeLeRe

Usually, FeLeRe can be used in either of the following ways:

* Testing optimization methods for robustness in typical federated learning conditions (client lag, heterogenous data) for proof-of-concept purposes
* Comparing different methods against each other

### Main components

At a granular level, FeLeRe is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **felere** | Federated learning library for research with simulated client-server interactions. |
| **felere.pipeline** | Algorithm testing and analysis management |
|**felere.function**| Customizable or pre-implemented functions to be optimized |
|**felere.optimization**| Customizable or pre-implemented optimization methods |
| **felere.common.distributor** | Data ditribution for federated learning clients, considering heterogeneity |
| **felere.common.simulation** | Client-server communication simulation, client sampling and model updating|
| **felere.common.datasets** | Easy-to-load pre-defined datasets for convenience |

## Example of usage

![readme-pipeline](./res/readme/readme-pipeline.gif)
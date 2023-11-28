from common.decorators import singleton
from common.sampler import ISampler

from simulation.common.timer import Timer
from simulation.web.client import Client
from simulation.web.server import Server

from typing import Dict

@singleton
class Simulation:
  _server: Server
  _clients: Dict[str, Client]

  def __init__(self, sampler: ISampler = None):
    if sampler is not None:
      self._server = Server(timer=Timer(sampler=sampler))
    else:
      self._server = Server()
    self._clients = {}
  
  @property
  def server(self):
    return self._server

  @property
  def client(self, name) -> Client:
    return self._clients[name]
  
  def create_client(self, name: str) -> Client:
    client = Client(self.server.create_pipe(name=name))
    self._clients[name] = client

    return client



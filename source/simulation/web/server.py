from web.api import ISerializable, IServer
from simulation.web.pipe import Event, Pipe
from random import sample


from typing import List, Dict, Set
from time import time


class SimulationServer(IServer):
  def __init__(self, timer=time):
    self.pipe_to: Dict[str, Pipe] = {} 
    self.pipe_from: Dict[str, Pipe] = {}
    self.timer = timer
  
  def create_pipe(self, name: str) -> [Pipe]:
    """
      [`send`, `receive`]
    """
    self.pipe_to[name] = Pipe()
    self.pipe_from[name] = Pipe()

    return [self.pipe_from[name], self.pipe_to[name]]

  def send(
    self,
    event: Event,
    receivers: List[str] | int = None,
    timeout: float = 5
  ) -> int:
    if len(self.pipe_to) == 0:
      return
    
    target: List[Pipe] = []
    if receivers is not None:
      raise ValueError
    else:
      target = self.pipe_to.values()

    print(f"{len(target)=}")
    for client in target:
      client.put(event)

    return True
  
  def receive(
    self,
    senders_count: int = None,
    timeout: float = 5,
    blocking_wait=False
  ) -> Dict[str, Event]:
    if len(self.pipe_from) == 0:
      return {}
    
    deadline = self.timer() + timeout

    received: Dict[str, Event] = {}
    senders = set(self.pipe_from.keys())
    while len(senders) > 0 and len(received) < senders_count:
      done: Set[str] = set()
      for sender in senders:
        if self.pipe_from[sender].empty():
          continue

        got = self.pipe_from[sender].get(block=blocking_wait)
        if got.time > deadline:
          self.pipe_from[sender].put(got)
          done.add(sender)
        else:
          received[sender] = got
      
      for sender in done:
        senders.remove(sender)
    
    return received
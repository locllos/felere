from web.api import ISerializable, IClient
from source.simulation.common.pipe import Pipe, PipePair, Event

from typing import List, Dict

class SimulationClient(IClient):
  def __init__(
    self,
    pair: PipePair
  ):
    self.pipe_to: Pipe = pair.pipe_to
    self.pipe_from: Pipe = pair.pipe_from

  def send(
    self,
    data: Event,
    timeout: float = 5
  ) -> bool:
    self.pipe_to.put(data)

    return True

  def receive(
      self, 
      timeout: float = 5,
      blocking_wait=False
  ) -> Dict[str, Event]:
    while blocking_wait is False and \
          self.pipe_from.empty():
      continue
    
    return self.pipe_from.get(block=blocking_wait)
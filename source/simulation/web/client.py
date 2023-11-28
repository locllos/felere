from common.containers import Data
from simulation.common.timer import simulation_timer
from web.api import IClient
from common.containers import ISerializable
from simulation.common.pipe import Pipe, PipePair, Event

from typing import List, Dict, Tuple

class Client(IClient):
  def __init__(
    self,
    pair: PipePair,
    timer = simulation_timer
  ):
    self.pipe_to: Pipe = pair.pipe_to
    self.pipe_from: Pipe = pair.pipe_from
    self.timer = timer

  def send(
    self,
    data: Data,
    timeout: float = 5
  ) -> bool:
    self.pipe_to.put(Event(
      data.serialize(),
      data.version,
      self.timer()
    ))

    return True

  def receive(
      self, 
      timeout: float = 5,
      blocking_wait=False,
      return_time=False
  ) -> Data | Tuple[Data, float]:
    deadline = self.timer() + timeout
    while blocking_wait is False and \
          self.pipe_from.empty():
      continue
    
    got = self.pipe_from.get(block=blocking_wait)
    if got.time > deadline:
      self.pipe_from.put(got)

    if return_time:
      return (Data.deserialize(got.data), got.time)

    return Data.deserialize(got.data)
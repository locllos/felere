from common.containers import Data
from simulation.common.timer import timer
from web.api import IClient
from common.containers import ISerializable
from simulation.common.pipe import Pipe, PipePair, Event
from common.logging import _logger

from typing import Tuple

class Client(IClient):
  def __init__(
    self,
    pair: PipePair,
    timer = timer
  ):
    self.pipe_to: Pipe = pair.pipe_to
    self.pipe_from: Pipe = pair.pipe_from
    self.timer = timer

  def send(
    self,
    data: Data,
    timeout: float = 5
  ) -> bool:
    now = self.timer()

    _logger.info(f"Send message[{now}|{data.version}] to server")
    self.pipe_to.put(Event(
      data.serialize(),
      data.version,
      now
    ))

    return True

  def receive(
      self, 
      timeout: float = 5,
      blocking_wait = False,
      return_time = False
  ) -> Data | Tuple[Data, float]:
    now = self.timer()
    deadline = now + timeout

    while blocking_wait is False and \
          self.pipe_from.empty():
      _logger.info(f"Client has not received messages from server.")
      continue
    
    got = self.pipe_from.get(block=blocking_wait)
    if got.time > deadline:
      self.pipe_from.put(got)
      _logger.info(
        f"Message from server has expired deadline: {got.time=}|{deadline=}. Send it back"
      )

    _logger.info(
      f"Message from server was successful received"
    )
    self.timer.speedup(on = got.time - now if got.time-now > 0 else 0)

    if return_time:
      return (Data.deserialize(got.data), got.time)

    return Data.deserialize(got.data)
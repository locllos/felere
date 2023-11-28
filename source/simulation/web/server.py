from web.api import IServer
from common.containers import Data
from simulation.common.pipe import Event, Pipe, PipePair
from simulation.common.timer import timer

from typing import List, Dict, Set, Tuple

from common.logging import _logger

class Server(IServer):
  def __init__(self, timer=timer):
    self.pipes_to: Dict[str, Pipe] = {} 
    self.pipes_from: Dict[str, Pipe] = {}
    self.timer = timer
  
  def create_pipe(self, name: str) -> PipePair:
    self.pipes_to[name] = Pipe()
    self.pipes_from[name] = Pipe()

    _logger.info(f"creating pipe for {name} and server")

    return PipePair(
      pipe_to=self.pipes_from[name],
      pipe_from=self.pipes_to[name]
    )

  def send(
    self,
    data: Data,
    receivers: List[str] | int = None,
    timeout: float = 5
  ) -> int:
    if len(self.pipes_to) == 0:
      _logger.info(f"Empty `pipes_to`")
      return False
    
    event = Event(
      data.serialize(),
      data.version,
      self.timer()
    )

    target: List[Pipe] = []
    if receivers is not None:
      raise ValueError
    else:
      target = self.pipes_to.values()

    _logger.info(f"Trying to send message to {target=}")
    for client in target:
      _logger.info(f"Sending message to {client=}")
      client.put(event)

    return True
  
  def receive(
    self,
    senders_count: int = None,
    timeout: float = 5,
    blocking_wait=False,
    return_time=False
  ) -> Dict[str, Data] | Tuple[Dict[str, Data], Dict[str, float]]:
    if len(self.pipes_from) == 0:
      _logger.info("Empty pipes from")
      return {}
    
    now = self.timer()
    deadline = now + timeout
    time_diff = 0

    received: Dict[str, Data] = {}
    times: Dict[str, float] = {}
    senders = set(self.pipes_from.keys())
    while len(senders) > 0 and \
          len(received) < senders_count:
      done: Set[str] = set()
      for sender in senders:
        _logger.info(f"Processing {sender}")
        if self.pipes_from[sender].empty():
          _logger.info(f"Server has not received messages from {sender=}.")
          # MAYBE: this is not neccessary
          # spinlock waiting on data to be sent
          continue

        got = self.pipes_from[sender].get(block=blocking_wait)
        _logger.info(
          f"Message[{got.time=}|{got.version=}] from {sender=}."
        )
        if got.time > deadline:
          _logger.info(
            f"Message from {sender=} has expired deadline: {got.time=}|{deadline=}. Send it back"
          )
          # send back to future :)
          # TODO: when we will use `version` 
          # we must drop such values if its version < servers version
          self.pipes_from[sender].put(got)
        else: 
          time_diff = max(
              time_diff,
              got.time - now if got.time - now > 0 else 0
          )
          if return_time:
            times[sender] = got.time

          received[sender] = Data.deserialize(got.data)
          _logger.info(
            f"Message from {sender=} was successful received"
          )

        done.add(sender)
      
      for sender in done:
        senders.remove(sender)
    
    self.timer.speedup(on=time_diff)
    return received, times
from web.api import ISerializable, IClient

from typing import List

class SimulationClient(IClient):
  def __init__(
    self  
  ):
    pass

  def send(
    data: ISerializable, 
    receivers: List[str] | int = [], 
    timeout: float = 5
  ) -> int:
    raise NotImplementedError
  
  def receive(
    senders: List[str] | int = [],
    timeout: float = 5
  ) -> List[ISerializable]:
    raise NotImplementedError
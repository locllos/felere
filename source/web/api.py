from typing import List, Dict

class IClient:
  def send(self, data, timeout: float = 5) -> bool:
    raise NotImplementedError
  def receive(self, timeout: float = 5):
    raise NotImplementedError

class IServer:
  def send(
    self,
    data, 
    receivers: List[str] | int = None, 
    timeout: float = 5
  ) -> int:
    raise NotImplementedError
  
  def receive(
    self,
    senders_count: int = None,
    timeout: float = 5
  ) -> Dict:
    raise NotImplementedError
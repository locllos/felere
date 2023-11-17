from typing import List

class ISerializable:
  def serialize():
    raise NotImplementedError
  def deserialize():
    raise NotImplementedError

class IClient:
  def send(data: ISerializable, timeout: float = 5) -> bool:
    raise NotImplementedError
  def receive(timeout: float = 5):
    raise NotImplementedError

class IServer:
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
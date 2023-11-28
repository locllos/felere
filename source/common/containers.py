from dataclasses import dataclass
import pickle

class ISerializable:
  def serialize(self) -> bytes:
    raise NotImplementedError
  
  @staticmethod
  def deserialize(data: bytes):
    raise NotImplementedError


@dataclass
class Data(ISerializable):
  version: int = 0
  string: str = ""

  def serialize(self) -> bytes:
    return pickle.dumps(self)
  
  @staticmethod
  def deserialize(data: bytes):
    return pickle.loads(data)
  

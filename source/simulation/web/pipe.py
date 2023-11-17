from web.api import ISerializable

from queue import PriorityQueue
from collections import namedtuple

from dataclasses import dataclass


@dataclass
class Event:
  time: float
  version: int
  data: ISerializable

  def __lt__(self, other):
    return self.version < other.version or \
           self.version == other.version and self.time < other.time


Pipe = PriorityQueue[Event]


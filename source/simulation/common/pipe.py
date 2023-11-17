from web.api import ISerializable

from queue import PriorityQueue
from time import time
from collections import namedtuple

from dataclasses import dataclass


@dataclass
class Event:
  data: bytes
  version: int # yes, this is not universal and some algorithmic logic appears at not algorithmic code
  time: float = time()

  def __lt__(self, other):
    return self.version < other.version or \
           self.version == other.version and self.time < other.time


Pipe = PriorityQueue[Event]
PipePair = namedtuple("PipePair", ["pipe_from", "pipe_to"])


from web.api import ISerializable

from queue import PriorityQueue
from dataclasses import dataclass
from typing import NewType


@dataclass
class EventStamp:
  timestamp: float
  version: int

Event = NewType("Event", tuple(EventStamp, ISerializable))

# TODO: impl this
class Pipe:
  def __init__(self):
    # notice that PriorityQueue is thread-safe
    self.queue = PriorityQueue()

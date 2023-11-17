# Now we are working on simluation. Skip real server-clients interaction for now

from dataclasses import dataclass
from queue import PriorityQueue

@dataclass
class EventStamp:
  timestamp: float
  version: int



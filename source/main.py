# Now we are working on simulation. Skip real server-clients interaction for now

from simulation.web.client import Client
from common.containers import Data
from random import randint, sample
from simulation.common.timer import timer
from simulation.general import Simulation

simulation = Simulation()

server = simulation.server

clients: list[Client] = [
  simulation.create_client(f"{i}") for i in range(25)
]

server.send(Data(
  version=randint(0, 25), 
  string=b"hello everyone!"
))


for client in clients:
  got, times = client.receive(return_time=True)
  print(got.version, times)

subrange = sample(range(25), 16)
for i in subrange:
  clients[i].send(Data(
    version=randint(0, 10),
    string=f"message for {i}"
  ))


messages, times = server.receive(15, return_time=True)
print(*[f"({i: >2}) [{key: >2}] = {value}\n" for i, (key, value) in enumerate(messages.items())])
print(*[f"({i: >2}) [{key: >2}] = {value}\n" for i, (key, value) in enumerate(times.items())])
print(f"current time: {timer()}")
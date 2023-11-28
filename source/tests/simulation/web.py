import unittest
from random import randint, sample

from simulation.common.timer import timer
from simulation.general import Simulation
from simulation.common.pipe import Data

class TestSimulation(unittest.TestCase):
  clients_count = 25
  subclients_count = 16

  def setUp(self):
    self.simulation = Simulation()
    self.server = self.simulation.server
    self.clients = [
      self.simulation.create_client(f"{i}") for i in range(25)
    ]

  def test_timer(self):
    current_time = timer()
    
    self.assertGreaterEqual(current_time, 0.)
    self.assertIsInstance(current_time, float)

  def test_client_receive(self):
    self.server.send(Data(
      version=randint(0, 25),
      string="hello everyone!"
    ))

    for client in self.clients:
      got = client.receive()
      self.assertIsNotNone(got)
      self.assertEqual(got.string, "hello everyone!")

  def test_server_receive(self):
    subrange = sample(range(self.clients_count), self.subclients_count)
    for i in subrange:
      self.clients[i].send(Data(
        version=randint(0, 10),
        string=f"message for {i}"
    ))

    messages = self.server.receive(self.subclients_count)
    self.assertIsInstance(messages, dict)
    self.assertEqual(len(messages), self.subclients_count)

    for i in subrange:
      self.assertEqual(
        f"message for {i}", 
        messages[f"{i}"].string
      )

if __name__ == '__main__':
  unittest.main()
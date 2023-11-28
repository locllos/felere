import unittest
from random import randint, sample

from simulation.common.timer import timer
from simulation.general import Simulation
from simulation.common.pipe import Data

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation()
        self.server = self.simulation.server
        self.clients = [self.simulation.create_client(f"{i}") for i in range(25)]

    def test_client_receive(self):
        self.server.send(Data(
            version=randint(0, 25),
            string=b"hello everyone!"
        ))

        for client in self.clients:
            got, times = client.receive(return_time=True)
            self.assertIsNotNone(got)
            self.assertIsInstance(times, float)

    def test_server_receive(self):
        subrange = sample(range(25), 16)
        for i in subrange:
            self.clients[i].send(Data(
                version=randint(0, 10),
                string=f"message for {i}"
            ))

        messages, times = self.server.receive(15, return_time=True)
        self.assertIsInstance(messages, dict)
        self.assertIsInstance(times, dict)
        self.assertEqual(len(messages), len(times))

    def test_timer(self):
        current_time = timer()
        self.assertIsInstance(current_time, float)

if __name__ == '__main__':
    unittest.main()
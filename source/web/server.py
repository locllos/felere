from common.serializer import SimpleSerializer
from common.io_routine import timeouted, fire_and_forget_async, fire_and_forget

import asyncio
import logging
import threading

from dataclasses import dataclass
from random import sample

from typing import List
from typing import Set
from typing import Dict

logging.basicConfig(level=logging.INFO, filename="server.log",filemode="w")


# Что хочу иметь

class TCPServer:
  @dataclass
  class Client:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

  def __init__(
    self,
    address: str = '127.0.0.1',
    port: int = 8888,
    clients_count: int = 0,
    read_separator: bytes = b'$',
    serializer: SimpleSerializer = SimpleSerializer()
  ):
    self.clients: dict[str, TCPServer.Client] = {}
    self.address: str = address
    self.port: int = port
    self.clients_count = clients_count
    self.read_separator = read_separator
    self.serializer = serializer
    # self.loop = asyncio.new_event_loop()
    self.server = None
    self.accept_connections = None

  async def start(self):
    async def startup():
      logging.info("Server is starting...")
      self.server = await asyncio.start_server(
        lambda reader, writer: self._handle_new_connection(reader, writer),
        host=self.address,
        port=self.port
      )
      logging.info("Server successfully started")
    
    asyncio.run(startup())

    async def accept_connections():
      async with self.server:
        logging.info("Loop successfully started")
        await self.server.serve_forever()
    
    self.accept_connections = asyncio.create_task(accept_connections())
    

  def is_started(self):
    return self.accept_connections is not None
  
  def stop(self):
    if self.is_started() is False:
      return False
    
    return self.accept_connections.cancel()
  
  
  async def send(self, message: bytes, receivers: List[str] | int = [], timeout: float = 5):
    """
      empty receivers means that we want to send message to all clients
    """
    if len(self.clients) == 0:
      return
    
    target: List[str] = []
    if type(receivers) is list and receivers == [] or \
       type(receivers) is int and receivers == 0:
      target = list(self.clients.keys())
    elif type(receivers) is list:
      target = receivers
    elif type(receivers) is int:
      target = sample(list(self.clients.keys()), receivers)
    else:
      raise ValueError
    
    print(target)
    tasks = []
    for _, writer in [(peername, self.clients.get(peername).writer) for peername in target]:
      writer.write(message)
      await writer
    

  async def receive(self, senders: List[str] | int = [], timeout: float = 5) -> Dict[str, bytes]:
    target: List[str] = []
    if type(senders) is List and senders is [] or \
       type(senders) is int and senders == 0:
      target = list(self.clients.keys())
    elif type(senders) is List:
      target = senders
    elif type(senders) is int:
      target = sample(list(self.clients.keys()), senders)
    else:
      raise ValueError
    
    tasks = []
    data: Dict[str, bytes]
    for peername, reader in [(peername, self.clients.get(peername).writer) for peername in target]:

      async def timeout_read():
        data: bytes
        async def read():
          data = reader.readuntil(self.read_separator)

        try:
          await asyncio.wait_for(read())
          return data
        except:
          return b''
      

  async def _handle_new_connection(
    self,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter
  ):
    peername = writer.get_extra_info('peername')
    
    self.clients[peername] = TCPServer.Client(reader, writer)
    logging.info(f"New connection: {peername}")









class Server:
  @dataclass
  class Client:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

  def __init__(
    self,
    address: str = '127.0.0.1',
    port: int = 8888,
    serializer: SimpleSerializer = SimpleSerializer()
  ):  
    self.address: str = address
    self.port: int = port
    self.is_closed: bool = False
    self.clients: List[Server.Client] = []
    self.serializer = serializer
    self.server = None

  def run(self):
    async def start():
      logging.info("Server is starting...")
      self.server = await asyncio.start_server(
        lambda reader, writer: self.handle_new_connection(reader, writer),
        host=self.address,
        port=self.port
      )
      logging.info("Server successfully started")
    
    async def loop():
      async with self.server:
        logging.info("Loop successfully started")
        await self.server.serve_forever()
    
    asyncio.run(start())

  async def handle_new_connection(
    self,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter
  ):
    peername = writer.get_extra_info('peername')
    logging.info(f"New connection: {peername}")

    while reader.at_eof() is False:
      await self.handle_connection(client=Server.Client(reader, writer))

    logging.info(f"Connection: {peername} was closed")
    writer.close()
    await writer.wait_closed()


  async def handle_connection(self, client: Client):
    peername = client.writer.get_extra_info('peername')

    try:
      logging.info(f"Awaiting read from {peername}")
      raw = await client.reader.readuntil(b"$")
    except:
      logging.warning(
        f"{peername} closed the connection while reading."
      )

    data = raw.decode("utf-8")
    logging.info(f"Data {data} by {peername}")

    try:
      client.writer.write(f"{data}".encode())
      await client.writer.drain()
    except:
      logging.warning(
        f"{peername} closed the connection while writing."
      )


# server = Server()
# server.run()

class TCPClient:
  def __init__(self, address: str, port: int):
    pass

  def Launch():
    pass

  async def launch():
    pass

  async def send(message):
    pass

  async def receive():
    pass
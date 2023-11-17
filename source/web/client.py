import asyncio
import logging
import sys

from dataclasses import dataclass
from typing import List

logging.basicConfig(level=logging.INFO, filename="client.log", filemode="w")

class Client:
  def __init__(
    self,
    address: str = '127.0.0.1',
    port: int = 8888
  ):
    self.address = address
    self.port = port
    self.is_closed = False


  async def interact(self):
    self.reader, self.writer = await asyncio.open_connection(self.address, self.port)

    if self.reader is None or self.writer is None:
      logging.warning(f"Unable to open a connection={self.address}:{self.port}")
      return
    else:
      logging.info(f"Successful connection={self.address}:{self.port}")


    got_line = input("Input: ")
    while self.reader.at_eof() is False and got_line != "$":
      await self.handle_input(got_line)
      got_line = input("Input: ")
    
    self.writer.close()
    await self.writer.wait_closed()


  async def handle_input(self, input: str):
    peer_name = self.writer.get_extra_info('peername')

    try:
      self.writer.write(f"{input}$".encode("utf-8"))
      await self.writer.drain()

    except (asyncio.exceptions.BrokenPipeError, ConnectionResetError):
      logging.warning(
        f"{peer_name} closed the connection while writing."
      )

    try:
      raw = await client.reader.readuntil(b"$")
    except:
      logging.warning(
        f"{peer_name} closed the connection while reading."
      )

    data = raw.decode()
    logging.info(f"Data: {data} by {peer_name}")


client = Client()
asyncio.run(client.interact())

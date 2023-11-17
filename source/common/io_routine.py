import asyncio
import logging


def fire_and_forget(f, loop = asyncio.get_event_loop()):
	def wrapped(*args, **kwargs):
		loop.run_in_executor(None, f, *args, *kwargs)

	return wrapped

def fire_and_forget_async(func, loop=asyncio.get_event_loop()):
  @fire_and_forget
  def wrapped(*args, **kwargs):
    return loop.run_until_complete(func(*args, *kwargs))
    
  return wrapped

def timeouted(f, message, timeout=5, exception_message = None, exception_list=[]):
	async def wrapped(*args, **kwargs):
		try:
			await asyncio.wait_for(f(*args, *kwargs), timeout)
			return message    
		except *exception_list:
			logging.error("\n".join([
				f"Exception: {type(exception).__name__}. Args: {exception.args}." 
				for exception in exception_list
			]))
		return exception_message
    
	return wrapped



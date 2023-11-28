import logging, sys

from common.const import kLogToFile

def _init_logger(stream=sys.stderr):
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[time=%(asctime)s]::[level=%(levelname)8s]::[path=%(pathname)s:%(lineno)d]::[thread=%(thread)d]::[message=%(message)s]")
    
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

_logger = None
def InitLogger():
  global _logger
  if "pytest" in sys.modules:
    _init_logger()
  else:
    _init_logger(open("logs/main.log", mode="w"))
  
  _logger = logging.getLogger("app")

InitLogger()

    
   
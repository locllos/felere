import logging, sys

def _init_logger(stream=sys.stderr):
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[time:%(asctime)s]::[level:%(levelname)8s]::[path=%(pathname)s:%(lineno)d]::[thread=%(thread)d]::[message=%(message)s]")
    
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

kLogToFile = True
if kLogToFile:
  _init_logger(open("logs/main.log", mode="w"))
else:
  _init_logger()
   
_logger = logging.getLogger("app")
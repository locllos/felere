import logging, sys

def _init_logger(stream = sys.stderr):
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s")
    handler = logging.StreamHandler(stream) \
                     .setFormatter(formatter)
    logger.addHandler(handler)

kLogToFile = True
if kLogToFile:
  _init_logger(open("../res/logs.log", mode="w"))
else:
  _init_logger()
   
_logger = logging.getLogger("app")
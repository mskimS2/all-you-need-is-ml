import os

try:
    import logging
    from loguru import logger    
except ImportError:
    os.system("pip install loguru")
    import logging
    from loguru import logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    
    logger = logging.getLogger("logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
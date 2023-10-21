import os

try:
    from loguru import logger
    
except ImportError:
    os.system("pip install loguru")
    
    import logging
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    
    logger = logging.getLogger("XGBoost")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
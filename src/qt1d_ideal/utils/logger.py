import logging
from pathlib import Path
from datetime import datetime

class SimulationLogger:
    def __init__(self, scenario_name, log_dir="logs", verbose=True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{scenario_name}_{timestamp}.log"
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(f"qt1d_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def log_parameters(self, params):
        self.info(f"Parameters: {params}")
    
    def log_timing(self, timing):
        self.info(f"Timing: {timing}")
    
    def finalize(self):
        self.info(f"Completed: {self.scenario_name}")

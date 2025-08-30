from loguru import logger
from pathlib import Path

def setup_logger(log_dir: str = "models") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(Path(log_dir) / "run.log", rotation="1 MB", retention=5)
    logger.info("Logger initialized")
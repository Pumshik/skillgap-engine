import sys
from loguru import logger
from pathlib import Path

def setup_logger(log_file: str = "artifacts/logs/app.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(log_file, rotation="10 MB", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
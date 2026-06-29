import os
import logging
from loguru import logger
from src.config import LOG_DIR, OUTPUT_DIR, MODEL_DIR

def setup_dirs():
    """Create all required directories."""
    for d in [LOG_DIR, OUTPUT_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

def get_logger(name: str):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    logger.add(log_path, rotation="5 MB", level="INFO")
    return logger

def save_figure(fig, filename: str):
    """Save matplotlib figure to outputs/."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    logger.info(f"Figure saved → {path}")
    return path
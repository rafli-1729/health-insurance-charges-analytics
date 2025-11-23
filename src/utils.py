# src/utils.py
import logging, random, tf, os
import numpy as np
from typing import Iterable, List

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        print("Warning: tf.config.experimental.enable_op_determinism() tidak tersedia di versi TF ini.")

# Basic logger used across modules
def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = get_logger(__name__)

def clean_feature_names(raw_names: Iterable[str]) -> List[str]:
    """
    Remove common prefixes inserted by ColumnTransformer / Pipelines.
    Keeps the last token after double-underscore(s).
    """
    cleaned = [str(n).split("__")[-1] for n in raw_names]
    return cleaned

def safe_assert_sufficient_rows(n_rows: int, n_splits: int):
    """
    Raise informative error if data too small for CV.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if n_rows < n_splits:
        raise ValueError(f"Number of rows ({n_rows}) is smaller than n_splits ({n_splits}). "
                         "Reduce n_splits or provide more data.")

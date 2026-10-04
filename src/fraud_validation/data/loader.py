"""Input helpers for historical transaction datasets."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str | Path, **read_csv_options: Any) -> pd.DataFrame:
    """Load a CSV without imposing a dataset-specific schema."""
    return pd.read_csv(path, **read_csv_options)
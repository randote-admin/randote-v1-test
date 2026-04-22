import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

def train_val_split(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - val_ratio))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]

def tokenize_batch(texts: list, tokenizer, max_len: int = 512) -> dict:
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

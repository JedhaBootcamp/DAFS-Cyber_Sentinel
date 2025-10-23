from __future__ import annotations
import pandas as pd

# Expected categorical columns in your sample
EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
# Columns that are 0/1 flags in KDD-like data (if present)
EXPECTED_BOOLISH = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login"
]
TARGET = "class"   # "normal"/"anomaly" or 0/1


def _coerce_booleans(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            # Many datasets already use 0/1; this is idempotent
            df[c] = df[c].astype("float").fillna(0).astype("int64")


def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and normalize target to 0/1 (1 = anomaly).
    - Strips whitespace from string columns to reduce cardinality noise
    - Coerces common boolean-ish columns to 0/1 (if present)
    """
    df = pd.read_csv(path)

    # Basic schema checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET}'")

    # Normalize target → 0/1
    if df[TARGET].dtype == object:
        df[TARGET] = (
            df[TARGET]
            .astype(str).str.strip().str.lower()
            .map({"normal": 0, "anomaly": 1, "attack": 1})
        )
    df[TARGET] = df[TARGET].astype("int64")

    # Clean strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Coerce obvious boolean-ish columns
    _coerce_booleans(df, EXPECTED_BOOLISH)

    return df

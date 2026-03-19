import pandas as pd

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    TODO (Student A): add dtype hints, NA handling policy, schema validation, and logging.
    """
    df = pd.read_csv(path)

    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    raw = df[TARGET_COL].astype(str).str.strip().str.lower()

    # case 1: text labels normal/anomaly
    if set(raw.unique()).issubset({"normal", "anomaly"}):
        mapping = {"anomaly": 1, "normal": 0}
        df[TARGET_COL] = raw.map(mapping)

    # case 2: already numeric 0/1 stored as text
    elif set(raw.unique()).issubset({"0", "1"}):
        df[TARGET_COL] = raw.astype(int)

    else:
        raise ValueError(
            f"Unexpected target values in '{TARGET_COL}': {raw.unique().tolist()}"
        )

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(
            f"Target must be 'normal'/'anomaly' or 0/1 after mapping. "
            f"Found: {df[TARGET_COL].unique().tolist()}"
        )

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df

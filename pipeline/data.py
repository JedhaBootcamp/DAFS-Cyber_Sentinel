import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/data.log"),logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"] #string, string, string
TARGET_COL = "class"   # values: "normal" / "anomaly"

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    TODO (Student A): 
    - add dtype hints, 
    - NA handling policy, 
    - schema validation,
    - logging. : print for console logging ; use logging for log in a log file ?
    """
    data_types = {"protocol_type": str, "service": str, "flag" : str, "class" : str}
    logger.info("Importing dataset into Dataframe...")
    try:
        df = pd.read_csv(path, dtype=data_types)
    except FileNotFoundError:
            logger.error("The file {path} was not found")
            #raise FileNotFoundError(f"The file {path} was not found")
    except ValueError:
        logger.error("The data types expected for the dataframe don't match.")    
    else:
        logger.info(f"Dataset {path} successfully imported as dataframe")

    # basic sanity checks
    #print("Checking for missing columns...")
    logger.info("Checking for missing columns...")
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        logger.error(f"CSV missing expected categorical columns: {missing}")
        #raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        #raise ValueError(f"CSV missing target column '{TARGET_COL}'")
        logger.error(f"CSV missing target column '{TARGET_COL}'")
    else:
        logger.info("No missing columns.")

    #dtypes check
    logger.info("Validating target column...")
    # Normalize target → binary 0/1
    mapping = {"anomaly": 1, "normal": 0}
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map(mapping)
    # If already numeric, assume 0/1
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        logger.error("Target must be 'normal'/'anomaly' or 0/1 after mapping.")
        #raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")
    else:
        logger.info("The target column was successfully normalized")

    return df

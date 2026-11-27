import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split

from .config import DATA_FILE

PROPOSER = ["Johannes", "Jonas", "Florian"]
STATUS = ["Accepted", "Denied"]

rng = np.random.default_rng(42)

def make_data_split_fixed_y(d:float, l: int = 10) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Makes X_train, X_test, y_train, y_test
    for fixed value of d

    Args:
        d (float): value of all y 
        d (int): amount of rows (80/20% split, so 10 rows = 8 train 2 test)
        
    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]: X_train, X_test, y_train, y_test
    """
    train = l//5 * 4
    X = np.array([[float(i)] for i in range(10)], dtype=np.float64)
    y = np.array([d for _ in range(10)], dtype=np.float64)
    return X[:train], X[train:], y[:train], y[train:]


def make_data_set(size: int = 250, categories: list[str] = PROPOSER):
    """Generate the data.csv file

    Args:
        size (int, optional): size of dataset. Defaults to 100.
        categories (list[str], optional): list of possible categories. Defaults to WRITER list.
    """
    data ={
        "proposer": rng.choice(categories, size),
        "budget": rng.normal(100, 20, size)
    }
    acceptance = [
        STATUS[0] if
        writer==PROPOSER[0] and
        data["budget"][i] < 100 and
        rng.uniform(0, 10) > 2.5
        else STATUS[1]
        for i, writer in enumerate(data["proposer"])
    ]
    
    data["status"] = acceptance
    DATA_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(data).to_csv(DATA_FILE, index=False)
    
def split_data_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        df.drop("status", axis=1),
        df["status"],
        test_size = 0.2,
        stratify=df["status"]
    )
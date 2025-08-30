from sklearn.datasets import load_iris
import pandas as pd

def load_dataset(name: str = "iris") -> tuple[pd.DataFrame, pd.Series]:
    if name == "iris":
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        return X, y
    raise ValueError(f"Unsupported dataset: {name}")
import gower
import pandas as pd
import numpy as np

def get_gower(categorical_cols : list[bool], df : pd.DataFrame) -> np.array:
    gower_matrix = gower.gower_matrix(df, cat_features = categorical_cols)
    return gower_matrix
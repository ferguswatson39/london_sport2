import gower
import pandas as pd
import numpy as np

def get_gower(categorical_cols : list[str], df : pd.DataFrame) -> np.array:
    cat_bool_mask = np.array([True if c in categorical_cols else False for c in df.columns])
    gower_matrix = gower.gower_matrix(df, cat_features = cat_bool_mask)
    for idx, value in enumerate(df.columns):
        print(f'{idx} | {cat_bool_mask[idx]} | {value}')
    return gower_matrix
import pandas as pd
import numpy as np

def perturb(df : pd.DataFrame, perturb_col : str) -> pd.DataFrame:
    """ 
    Perturbation function
    Takes perturb_col as argument and creates new col with perturbed inputs
    For obs which are already at the max, replaces these obs with NAN as to prevent impossible responses
    """
    df = df.copy()
    max_val = df[perturb_col].max()
    p_name = f'PERTURBED_{perturb_col}'
    df[p_name] = np.where(df[perturb_col] < max_val, df[perturb_col] + 1, np.nan)
    return df
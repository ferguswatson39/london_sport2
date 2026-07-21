import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue

def run_perturbation(df : pd.DataFrame, model : object, scaler : object):
    """
    Perturbation Function.

    Requires, df, model and scaler objects.

    Returns full df with original data and predictions in terms of probability for each var in to_perturb 

    """
    output_df = df.copy()
    dc = DataCatalogue()
    to_perturb = dc.get_perturbation_vars()
    print(to_perturb)
    for var in to_perturb:
        
        new = df.copy()
        if var not in new.columns:
            print(f'{var} not in DataFrame pre perturbation')
            continue
        old_var_name = f'PREDS_{var}'
        new_var_name = f'PERTURBED_PREDS_{var}'
        difference = f'DIFFERENCE_{var}'

        var_change = dc.get_perturbation_change(var)
        var_max = dc.get_perturbation_max(var)
        var_min = dc.get_perturbation_min(var)
        var_values = new[var].values
        var_idx = new.columns.get_loc(var)
        
        if var_change < 0:
            mask = np.where(var_values > var_min, True, False)
        elif var_change > 0:
            mask = np.where(var_values < var_max, True, False)

        X_scaled = scaler.transform(new[mask])
        non_p_preds = model.get_proba(X_scaled)
        X_scaled[:, var_idx] = X_scaled[:, var_idx] + var_change
        p_preds = model.get_proba(X_scaled)

        output_df.loc[mask, old_var_name] = non_p_preds
        output_df.loc[mask, new_var_name] = p_preds
        output_df.loc[mask, difference] = p_preds - non_p_preds
    return output_df

def create_and_join_diff_series(df : pd.DataFrame):
    series = []
    cols = [col for col in df.columns if 'DIFFERENCE' in col]
    print(cols)
    for c in cols:
       x = df.groupby('labels')[c].mean()
       series.append(x)
    return pd.concat(series, axis=1).reset_index()

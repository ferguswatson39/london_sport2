import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.perturbation.perturb_df import perturb
from sklearn.preprocessing import StandardScaler

def run_perturbation(df : pd.DataFrame, to_perturb : list, model : object, scaler : object):
    """
    Run perturbations function.
    Requires model object to have get_proba() method -- Ensure that models used in perturbation have .get_proba() method
    Creates two copies of df and uses one to store results and the other to compute perturbations
    Builds inverted boolean mask to filter data and ensure that samples which have max value arent perturbed to increase beyond limit.
    Saved output to new df
    """
    # using one df to store the output and one to do the perturbations
    output_df = df.copy()
    for var in to_perturb:
        new = df.copy()
        old_var_name = f'PREDS_{var}'
        new_var_name = f'PERTURBED_PREDS_{var}'
        difference = f'DIFFERENCE_{var}'

        perturbed_vals = perturb(new, var)
        # Use ~ to invert condition thus returns True for int values and False for NaN
        #https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
        nan_mask = ~np.isnan(perturbed_vals)
        X_scaled = scaler.transform(new[nan_mask])
        ## Generate the preds for non perturbed subset
        non_p_preds = model.get_proba(X_scaled)

        new.loc[nan_mask, var] = perturbed_vals[nan_mask]
        new_X_scaled = scaler.transform(new[nan_mask])
        p_preds = model.get_proba(new_X_scaled)
        
        output_df.loc[nan_mask, old_var_name] = non_p_preds
        output_df.loc[nan_mask, new_var_name] = p_preds
        output_df[difference] = (output_df[new_var_name] - output_df[old_var_name])
    return output_df
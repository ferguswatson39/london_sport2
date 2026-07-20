import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.perturbation.perturb_df import perturb

def run_perturbation(df : pd.DataFrame, to_perturb : list, model : object):
    """
    Run perturbations function.
    Requires model object to have scikit-learn predict_proba() method
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

        perturbed_vals = perturb(new, var)
        # Use ~ to invert condition thus returns True for int values and False for NaN
        #https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
        nan_mask = ~np.isnan(perturbed_vals)
        
        ## Generate the preds for non perturbed subset
        non_p_preds = model.predict_proba(new[nan_mask].values)

        new.loc[nan_mask, var] = perturbed_vals[nan_mask]

        p_preds = model.predict_proba(new[nan_mask].values)
        
        output_df.loc[nan_mask, old_var_name] = non_p_preds[: , 1]
        output_df.loc[nan_mask, new_var_name] = p_preds[: , 1]
    return output_df
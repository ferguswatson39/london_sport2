import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_clean_2022, one_hot_encode_frame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perturbation.models.logistic import Logistic


def create_perturbations(df : pd.DataFrame, model : object, scaler : object):
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


def create_and_join_diff_series(df : pd.DataFrame, group_by : str):
    series = []
    cols = [col for col in df.columns if 'DIFFERENCE' in col]
    for c in cols:
       # where group_by is either cluster 'labels' or borough code ect
       x = df.groupby(group_by)[c].mean()
       series.append(x)
    return pd.concat(series, axis=1).reset_index()

def run_perturbations(model : object, group_by : str):
    """
    Purturbation Pipeline Function

    Collects data from get_clean_2022 and implements one at a time sensitivity analysis following a 1 std change.

    Returns the full perturbation dataset and also the grouped series of average changes by group_by.

    For heatplot, use breakdowns.drop(columns='labels') then sns.heatmap....

    if y != 'MEMS7_ALL' or y != 'active':
        raise ValueError(f'{y} is not MEMS7_ALL or active')
    
    """
    # Get data and encode categoricals
    df = get_clean_2022()
    print(f'These are the vars:\n{df.columns}')
    df_encoded = one_hot_encode_frame(df)

    drop_cols = ['serial', 'year', 'LCA_Class', 'MEMS7_ALL', 'active']
    keep_vars = [var for var in df_encoded.columns if var not in drop_cols]
    
    labels = df_encoded['LCA_Class']
    Y = df_encoded['active']
    X = df_encoded[keep_vars]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, 'LCA_Class']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, Y_train)
    
    perturbed_df = create_perturbations(X_test, model, scaler)

    perturbed_df['labels'] = test_set_clusters
    breakdowns = create_and_join_diff_series(perturbed_df, group_by)
    return perturbed_df, breakdowns

# df, series = run_perturbations(group_by='labels')
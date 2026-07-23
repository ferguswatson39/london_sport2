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
from perturbation.models.classifiers.logistic import Logistic


def create_perturbations(df : pd.DataFrame, model : object, scaler : object, Y_test : pd.DataFrame):
    """
    Perturbation Function.

    Requires, df, model, scaler objects and specification of target var

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

        X_scaled = scale_data(scaler, model, new[mask], fit_transform = False)

        # We only want to save the accuracy or mse metrics for the non perturbed set
        # So use save_metric = True only for the first prediction generation
        non_p_preds = model.get_preds(X_scaled, Y_test, save_metric=True)
        X_scaled[:, var_idx] = X_scaled[:, var_idx] + var_change
        p_preds = model.get_preds(X_scaled, Y_test, save_metric=False)

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

def run_perturbations(df : pd.DataFrame, model : object, target : str, group_by : str, cluster_col : str):
    """
    Purturbation Pipeline Function

    Collects data from get_clean_2022 and implements one at a time sensitivity analysis following a 1 std change.

    Returns the full perturbation dataset and also the grouped series of average changes by group_by.

    For heatplot, use breakdowns.drop(columns='labels') then sns.heatmap....

    if y != 'MEMS7_ALL' or y != 'active':
        raise ValueError(f'{y} is not MEMS7_ALL or active')
    clusters: ['LCA_Class'..... names of other cols whcih are attatched to clusters]
    target: ['MEMS7_ALL', 'active']
    """

    print(f'These are the vars:\n{df.columns}')
    df_encoded = one_hot_encode_frame(df)

    drop_cols = ['serial', 'year', cluster_col, 'MEMS7_ALL', 'active']
    keep_vars = [var for var in df_encoded.columns if var not in drop_cols]
    
    labels = df_encoded[cluster_col]
    Y = df_encoded[target]
    X = df_encoded[keep_vars]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_col]

    scaler = StandardScaler()
    X_train_scaled = scale_data(scaler, model, X_train, fit_transform = True)
    
    model.fit(X_train_scaled, Y_train)
    
    perturbed_df = create_perturbations(X_test, model, scaler, Y_test)

    perturbed_df['labels'] = test_set_clusters
    breakdowns = create_and_join_diff_series(perturbed_df, group_by)
    return perturbed_df, breakdowns

#################
##### FIX #######
#################
# Perturbation relies on scaling data so its important that all data is scaled and 1sdev added to each var

def scale_data(scaler : object, model : object, X_train : pd.DataFrame, fit_transform : bool):
    """
    Tree models dont require data to be scaled.

    Function returns data as is for tree models and scales other models.

    Also if fit_transform = False, function assumes that scaler has already been fit and only applies transformation
    
    """
    trees = ['RFClassifier', 'LightGBMClassifier', 'RFRegressor', 'LightGBMRegressor']
    name = model.__class__.__name__
    if name in trees:
        return X_train
    if fit_transform:
        return scaler.fit_transform(X_train)
    return scaler.transform(X_train)


# df = get_clean_2022()
# df, series = run_perturbations(group_by='labels')
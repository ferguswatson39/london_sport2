import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_clean_2022, one_hot_encode_frame
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
from perturbation.models.classifiers.logistic import Logistic
from perturbation.models.classifiers.lgbm_classifier import LightGBMClassifier
from perturbation.models.classifiers.rf_classifier import RFClassifier
from perturbation.models.regressors.OLS_regressor import OLSRegressor
from perturbation.models.regressors.lgbm_regressor import LightGBMRegressor
from perturbation.models.regressors.rf_regressor import RFRegressor
from perturbation.models.classifiers.xgboost_classifier import XGBoostClassifier


best_case = {'model' : RFClassifier(),  }

def create_save_heatplot(breakdowns : pd.DataFrame, target : str, heatplot_name : str):
    heatplot_path = ROOT / 'figures' / 'perturbation'
    if target == 'active':
        title = 'Average Change in the Probability of Participating in Over 150 Minutes of Moderate Activity Per Week'
    else:
        title = 'Average Change in the Minutes of Moderate Activity per week'

    heat = breakdowns.drop(columns = 'labels')
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        heat,
        annot=True,
        cmap="YlOrBr",
        fmt='.2f'
    )
    plt.title(title)
    plt.ylabel('Clusters')
    plt.savefig(heatplot_path / heatplot_name)
    print(f'Saved hatplot figure to: {heatplot_path / heatplot_name}')

def create_and_join_diff_series(df : pd.DataFrame, group_by : str):
    series = []
    cols = [col for col in df.columns if 'DIFFERENCE' in col]
    for c in cols:
       # where group_by is either cluster 'labels' or borough code ect
       x = df.groupby(group_by)[c].mean()
       series.append(x)
    return pd.concat(series, axis=1).reset_index()

def get_one_class_preds(preds : np.array, class_int : int) -> np.array:
    if class_int not in [0, 1, 2]:
        raise ValueError(f'{class_int} not in [0, 1, 2] ')
    return preds[: , class_int]

def create_perturbations(X_test : pd.DataFrame, Y_test : pd.DataFrame, model : object, scaler : object, to_perturb_vars : list[str], dc : object = DataCatalogue()):
    """
    Perturbation Function.

    Requires, X_test, Y_test, model, dataCatalogue objects and to_perturb var list

    Returns full df with original data and predictions in terms of probability for each var in to_perturb 

    """
    output_df = X_test.copy()

    for var in to_perturb_vars:
        
        df = X_test.copy()
        if var not in df.columns:
            print(f'{var} not in DataFrame pre perturbation')
            continue

        var_change = dc.get_perturbation_change(var)
        var_max = dc.get_perturbation_max(var)
        var_min = dc.get_perturbation_min(var)

        var_values = df[var].values

        
        if var_change < 0:
            mask = np.where(var_values > var_min, True, False)
        elif var_change > 0:
            mask = np.where(var_values < var_max, True, False)

        X_masked = df.loc[mask].copy()

        Y_test_masked = Y_test[mask]

        X_masked_scaled = scaler.transform(X_masked)

        # We only want to save the accuracy or mse metrics for the non perturbed set
        # So use save_metric = True only for the first prediction generation -- The models metrics are generated when training so no need to save 
        non_p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric=False)

        var_idx = X_masked.columns.get_loc(var)
        X_masked_scaled[:,var_idx] = X_masked_scaled[:,var_idx] + var_change

        p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric=False)

        predictions_c0 = f'PREDS_C0_{var}'
        predictions_c1 = f'PREDS_C1_{var}'
        predictions_c2 = f'PREDS_C2_{var}'

        perturbed_predictions_c0 = f'PERTURBED_PREDS_C0_{var}'
        perturbed_predictions_c1 = f'PERTURBED_PREDS_C1_{var}'
        perturbed_predictions_c2 = f'PERTURBED_PREDS_C2_{var}'

        difference_c0 = f'DIFFERENCE_C0_{var}'
        difference_c1 = f'DIFFERENCE_C1_{var}'
        difference_c2 = f'DIFFERENCE_C2_{var}'

        output_df.loc[mask, predictions_c0] = get_one_class_preds(non_p_preds, 0)
        output_df.loc[mask, predictions_c1] = get_one_class_preds(non_p_preds, 1)
        output_df.loc[mask, predictions_c2] = get_one_class_preds(non_p_preds, 2)

        output_df.loc[mask, perturbed_predictions_c0] = get_one_class_preds(p_preds, 0)
        output_df.loc[mask, perturbed_predictions_c1] = get_one_class_preds(p_preds, 1)
        output_df.loc[mask, perturbed_predictions_c2] = get_one_class_preds(p_preds, 2)

        output_df.loc[mask, difference_c0] = get_one_class_preds(p_preds, 0) - get_one_class_preds(non_p_preds, 0)
        output_df.loc[mask, difference_c1] = get_one_class_preds(p_preds, 1) - get_one_class_preds(non_p_preds, 1)
        output_df.loc[mask, difference_c2] = get_one_class_preds(p_preds, 2) - get_one_class_preds(non_p_preds, 2)

    return output_df





if __name__ == '__main__':
    # Can edit run cases here to change run cases

    model_path = r'' # BEST MODEL PATH
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    scaler = model.get_scaler()

    target = 'active_status'
    group_by = 'labels'
    dc = DataCatalogue()
    to_perturb = dc.get_perturbation_vars()
    save_path = ROOT / 'results' / 'perturbation'

    X_test = pd.read_csv(ROOT / 'data' / 'perturbation' / 'X_test.csv')
    Y_test = pd.read_csv(ROOT / 'data' / 'perturbation' / 'Y_test.csv')
    test_set_clusters = pd.read_csv(ROOT / 'data' / 'perturbation' / 'test_set_clusters.csv')# verify that lca_class is correct here

    perturbations = create_perturbations(X_test, Y_test, model, scaler, to_perturb)


    perturbations[group_by] = test_set_clusters['LCA_Class'].values
    #breakdowns = create_and_join_diff_series(perturbations, group_by)

    df_name = f'{model.__class__.__name__}_perturbation_results_FINAL.csv'
    # heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot2.png'

    perturbations.to_csv(save_path / df_name, index = False)
    print(f'Saved pertubation results to: {save_path / df_name}')

    # create_save_heatplot(breakdowns, target, heatplot_name)
    print(f'Finished perturbaiton for {model.__class__.__name__}')





"""

def execute_perturbation_pipeline(df : pd.DataFrame, run_cases : dict):

    Main perturbation function.

    Iterates through run cases, optimises and then trains models. 

    Saves perturbed dfs and perturbed heatplots

    save_path = ROOT / 'results' / 'perturbation'

    dc = DataCatalogue()
    continuous_vars = dc.get_perturbation_core_contins()
    categoricals_to_encode = dc.get_perturbation_core_to_encode()# ['Gend3', 'Eth7', 'WorkStat8', 'HHLiv9'] doesnt include disab2

    print(f'Perturbation core continuous vars:\n>>> {continuous_vars}')

    for key, value in run_cases.items():
  
        model = run_cases[key]['model']
        target = run_cases[key]['target']

        group_by = 'labels'
        cluster_col = 'LCA_Class'

        df_name = f'{key}_{model.__class__.__name__}_perturbation_results2.csv'
        heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot2.png'

        print(f'Starting Perturbation for {model.__class__.__name__}')
        perturbed_df, breakdowns = build_perturbation_df(df, model, target, group_by, cluster_col, continuous_vars, categoricals_to_encode)

        perturbed_df.to_csv(save_path / df_name, index = False)
        print(f'Saved pertubation results to: {save_path / df_name}')

        create_save_heatplot(breakdowns, target, heatplot_name)
        print(f'Finished perturbaiton for {model.__class__.__name__}')


def build_perturbation_df(df : pd.DataFrame, model : object, target : str, group_by : str, cluster_col : str, continuous_vars : list[str], categoricals_to_encode : list[str]):

    Purturbation Pipeline Function

    Collects data from get_clean_2022 and implements one at a time sensitivity analysis following a 1 std change.

    Returns the full perturbation dataset and also the grouped series of average changes by group_by.

    For heatplot, use breakdowns.drop(columns='labels') then sns.heatmap....

    if y != 'MEMS7_ALL' or y != 'active':
        raise ValueError(f'{y} is not MEMS7_ALL or active')
    clusters: ['LCA_Class'..... names of other cols whcih are attatched to clusters]
    target: ['MEMS7_ALL', 'active']


    print(f'These are the vars:\n{df.columns}')

    if model.__class__.__name__ in ['OLSRegressor','Logistic']:
        drop = 'first'
    else:
        drop = None
    
    df_encoded = one_hot_encode_frame(df, categoricals_to_encode, drop)

    drop_cols = ['serial', 'year', cluster_col, 'MEMS7_ALL', 'active']
    keep_vars = [var for var in df_encoded.columns if var not in drop_cols]
    
    labels = df_encoded[cluster_col]
    Y = df_encoded[target]
    X = df_encoded[keep_vars] # Keep vars contains all of the vars in the original df + encoded category names - the drop cols bit

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_col]

    scaler = StandardScaler()

    X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars])

    #X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train, Y_train)
    
    perturbed_df = create_perturbations(X_test, Y_test, model, continuous_vars)

    # Saves model to pkl file 
    model.save_class()

    perturbed_df['labels'] = test_set_clusters
    breakdowns = create_and_join_diff_series(perturbed_df, group_by)
    return perturbed_df, breakdowns

    






if __name__ == '__main__':
    # Can edit run cases here to change run cases
    run_cases = classification_cases
    target = 'active_status'

    dc = DataCatalogue()
    df = get_clean_2022()
    df_copy = df.copy()

    to_perturb = dc.get_perturbation_vars()
    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()
    to_encode_vars = dc.get_perturbation_core_to_encode()

    group_by = 'labels'
    cluster_col = 'LCA_Class'
    save_path = ROOT / 'results' / 'perturbation'

    X_train, X_test, Y_train, Y_test, test_set_clusters = preprocess_perturbation(df_copy, cluster_col, target, to_encode_vars)

    for key, value in run_cases.items():

        run_num = key
        model = run_cases[key]['model']
        do_smote = run_cases[key]['smote']
        k_neigh = run_cases[key]['k_neighbors']
        strat = run_cases[key]['strategy']
        # target = run_cases[key]['target']

        if do_smote:
            print(f'Resampling with smote for run number: {run_num}')
            X_train, Y_train = smote_resample(X_train, Y_train, k_neigh, strat)
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, Y_train)


        perturbations = create_perturbations(X_test, Y_test, model, scaler, to_perturb)

        model.save_class()

        perturbations[group_by] = test_set_clusters
        #breakdowns = create_and_join_diff_series(perturbations, group_by)

        df_name = f'{key}_{model.__class__.__name__}_perturbation_results3.csv'
        # heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot2.png'

        perturbations.to_csv(save_path / df_name, index = False)
        print(f'Saved pertubation results to: {save_path / df_name}')

        # create_save_heatplot(breakdowns, target, heatplot_name)
        print(f'Finished perturbaiton for {model.__class__.__name__}')

"""
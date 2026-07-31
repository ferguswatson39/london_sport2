import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import seaborn as sns
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_clean_2022, one_hot_encode_frame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perturbation.models.classifiers.logistic import Logistic
from perturbation.models.classifiers.lgbm_classifier import LightGBMClassifier
from perturbation.models.classifiers.rf_classifier import RFClassifier
from perturbation.models.regressors.OLS_regressor import OLSRegressor
from perturbation.models.regressors.lgbm_regressor import LightGBMRegressor
from perturbation.models.regressors.rf_regressor import RFRegressor

run_cases = {
    0 : {'model' : Logistic(),'target' : 'active'},
    1 : {'model' : RFClassifier(),'target' : 'active'},
    2 : {'model' : LightGBMClassifier(),'target' : 'active'},
    3 : {'model' : OLSRegressor(), 'target' : 'MEMS7_ALL',},
    4 : {'model' : RFRegressor(), 'target' : 'MEMS7_ALL'},
    5 : {'model' : LightGBMRegressor(),'target' : 'MEMS7_ALL'}
} 
test_run_cases = {
    3 : {'model' : OLSRegressor(), 'target' : 'MEMS7_ALL',},
    4 : {'model' : RFRegressor(), 'target' : 'MEMS7_ALL'},
    5 : {'model' : LightGBMRegressor(),'target' : 'MEMS7_ALL'}
}

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
    print(df)
    print(df.columns)
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
        Y_test_masked = Y_test[mask]

        # We only want to save the accuracy or mse metrics for the non perturbed set
        # So use save_metric = True only for the first prediction generation
        non_p_preds = model.get_preds(X_scaled, Y_test_masked, save_metric=True)
        X_scaled[:, var_idx] = X_scaled[:, var_idx] + var_change
        p_preds = model.get_preds(X_scaled, Y_test_masked, save_metric=False)

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

def build_perturbation_df(df : pd.DataFrame, model : object, target : str, group_by : str, cluster_col : str, continuous_vars : list[str]):
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
    if model.__class__.__name__ in ['OLSRegressor','Logistic']:
        drop = 'first'
    else:
        drop = None
    
    df_encoded = one_hot_encode_frame(df, drop)

    drop_cols = ['serial', 'year', cluster_col, 'MEMS7_ALL', 'active']
    keep_vars = [var for var in df_encoded.columns if var not in drop_cols]
    
    labels = df_encoded[cluster_col]
    Y = df_encoded[target]
    X = df_encoded[keep_vars]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_col]

    scaler = StandardScaler()

    X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars])

    #X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train, Y_train)
    
    perturbed_df = create_perturbations(X_test, model, scaler, Y_test)

    # Saves model to pkl file 
    model.save_class()

    perturbed_df['labels'] = test_set_clusters
    breakdowns = create_and_join_diff_series(perturbed_df, group_by)
    return perturbed_df, breakdowns

def create_save_heatplot(breakdowns : pd.Series, target : str, heatplot_name : str):
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


def execute_perturbation_pipeline(df : pd.DataFrame, run_cases : dict):
    """
    Main perturbation function.

    Iterates through run cases, optimises and then trains models. 

    Saves perturbed dfs and perturbed heatplots
    """
    save_path = ROOT / 'results' / 'perturbation'
    dc = DataCatalogue()
    continuous_vars = dc.get_perturbation_contins()
    print(f'Perturbation continuous vars:\n>>> {continuous_vars}')

    for key, value in run_cases.items():
  
        model = run_cases[key]['model']
        target = run_cases[key]['target']

        group_by = 'labels'
        cluster_col = 'LCA_Class'

        df_name = f'{key}_{model.__class__.__name__}_perturbation_results2.csv'
        heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot2.png'

        print(f'Starting Perturbation for {model.__class__.__name__}')
        perturbed_df, breakdowns = build_perturbation_df(df, model, target, group_by, cluster_col, continuous_vars)

        perturbed_df.to_csv(save_path / df_name, index = False)
        print(f'Saved pertubation results to: {save_path / df_name}')

        create_save_heatplot(breakdowns, target, heatplot_name)
        print(f'Finished perturbaiton for {model.__class__.__name__}')


if __name__ == '__main__':
    df = get_clean_2022()
    df['MEMS7_ALL'] = df['MEMS7_ALL'].clip(upper=6720)
    df['MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    execute_perturbation_pipeline(df, test_run_cases)
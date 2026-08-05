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

tree_cases = {
    1 : {'model' : RFClassifier(),'target' : 'active'},
    2 : {'model' : LightGBMClassifier(),'target' : 'active'},
    4 : {'model' : RFRegressor(), 'target' : 'MEMS7_ALL'},
    5 : {'model' : LightGBMRegressor(),'target' : 'MEMS7_ALL'}
}

simple_cases = {
    0 : {'model' : Logistic(),'target' : 'active'},
    3 : {'model' : OLSRegressor(), 'target' : 'MEMS7_ALL',}
}

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

def create_perturbations(X_test : pd.DataFrame, Y_test : pd.DataFrame, model : object, scaler : object, dc : object, to_perturb_vars : list[str]):
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

        old_var_name = f'PREDS_{var}'
        new_var_name = f'PERTURBED_PREDS_{var}'
        difference = f'DIFFERENCE_{var}'

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
        # So use save_metric = True only for the first prediction generation
        non_p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric=True)

        var_idx = X_masked.columns.get_loc(var)
        X_masked_scaled[:,var_idx] = X_masked_scaled[:,var_idx] + var_change

        p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric=False)

        output_df.loc[mask, old_var_name] = non_p_preds
        output_df.loc[mask, new_var_name] = p_preds
        output_df.loc[mask, difference] = p_preds - non_p_preds
    return output_df


def preprocess_perturbation(df : pd.DataFrame, cluster_column : str, target : str, to_encode_vars : list[str]):


    df['MEMS7_ALL'] = df['MEMS7_ALL'].clip(upper=6720)
    df['MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    df['Disab2_POP'] = df['Disab2_POP'].replace({1.0 : 0.0, 2.0 : 1.0})

    assert to_encode_vars == ['Gend3', 'Eth7', 'WorkStat8', 'HHLiv9']

    df_encoded = one_hot_encode_frame(df, to_encode_vars, None)

    drop_cols = ['serial', 'year', 'LCA_Class', 'MEMS7_ALL', 'active']
    keep_cols = [col for col in df_encoded.columns if col not in drop_cols]
    print(f'Keep columns : {keep_cols}')

    labels = df_encoded[cluster_column]
    Y = df_encoded[target]
    X = df_encoded[keep_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_column]
    
    return X_train, X_test, Y_train, Y_test, test_set_clusters


if __name__ == '__main__':
    dc = DataCatalogue()
    df = get_clean_2022()
    df_copy = df.copy()

    to_perturb = dc.get_perturbation_vars()
    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()
    to_encode_vars = dc.get_perturbation_core_to_encode()

    group_by = 'labels'
    cluster_col = 'LCA_Class'
    save_path = ROOT / 'results' / 'perturbation'

    for key, value in run_cases.items():

        model = run_cases[key]['model']
        target = run_cases[key]['target']

        X_train, X_test, Y_train, Y_test, test_set_clusters = preprocess_perturbation(df_copy, cluster_col, target, to_encode_vars)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model.fit(X_train_scaled, Y_train)

        perturbations = create_perturbations(X_test, Y_test, model, scaler, dc, to_perturb)

        model.save_class()

        perturbations[group_by] = test_set_clusters
        breakdowns = create_and_join_diff_series(perturbations, group_by)

        df_name = f'{key}_{model.__class__.__name__}_perturbation_results2.csv'
        heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot2.png'

        perturbations.to_csv(save_path / df_name, index = False)
        print(f'Saved pertubation results to: {save_path / df_name}')

        create_save_heatplot(breakdowns, target, heatplot_name)
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

"""
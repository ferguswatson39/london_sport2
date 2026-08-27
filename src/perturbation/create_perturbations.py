import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.loading_data.data_catalogue import DataCatalogue


def create_save_heatplot(breakdowns : pd.DataFrame, heatplot_name : str):
    heatplot_path = ROOT / 'figures' / 'perturbation'

    heat = breakdowns.drop(columns = 'labels').T
    heat.columns = heat.columns + 1
    mask = heat.abs() < 0.03
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        heat,
        annot=True,
        #cmap="YlOrBr",
        cmap='coolwarm',
        fmt='.2f',
        mask=mask,
        center=0,
        yticklabels=[
            'Intrinsic Motivation', 'Extrinsic Motivation', 'Achieve Goals', 'Persistence',
            'Inclusive Area', 'See Similar People', 'Safe Excercise Spaces', 'Local Trust',
            'Neighborhood Cohesion', 'Anxiety', 'Happiness', 'Loneliness', 'Life Worthwhile'
        ],
        cbar=False

    )
    plt.xlabel('Clusters', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.savefig(heatplot_path / heatplot_name, bbox_inches='tight', dpi=500)
    print(f'Saved hatplot figure to: {heatplot_path / heatplot_name}')

def create_and_join_diff_series(df : pd.DataFrame, group_by : str):
    series = []
    cols = [col for col in df.columns if 'DIFFERENCE' in col]
    for c in cols:
       # where group_by is either cluster 'labels' or borough code ect
       x = df.groupby(group_by)[c].mean()
       series.append(x)
    return pd.concat(series, axis=1).reset_index()

def create_perturbations(X_test : pd.DataFrame, Y_test : pd.DataFrame, model : object, scaler : object, to_perturb_vars : list[str], dc : object = DataCatalogue()):
    """
    Perturbation Function.

    Requires, X_test, Y_test, model, dataCatalogue objects and to_perturb var list

    Returns full df with original data and predictions in terms of probability for each var in to_perturb 

    """
    output_df = X_test.copy()

    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()

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
        X_masked_scaled = X_masked.copy()

        Y_test_masked = Y_test[mask]

        X_masked_scaled[continuous_vars] = scaler.transform(X_masked_scaled[continuous_vars])

       # Accuracy metrics have already been saved during model fitting
        non_p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric = False)[: , 1]

        X_masked_scaled[var] = X_masked_scaled[var] + var_change
        p_preds = model.get_preds(X_masked_scaled, Y_test_masked, save_metric = False)[: , 1]

        predictions = f'PREDS_{var}'
        perturbed_predictions = f'PERTURBED_PREDS_{var}'
        difference = f'DIFFERENCE_{var}'

        output_df.loc[mask, predictions] = non_p_preds
        output_df.loc[mask, perturbed_predictions] = p_preds
        output_df.loc[mask, difference] = p_preds - non_p_preds
    
    return output_df

def check_perturbed_counts(df_path : Path, dc = DataCatalogue()):

    df = pd.read_csv(df_path)

    save_path = ROOT / 'results' / 'perturbation'
    file_name = f'{df_path.stem}_perturbation_counts.csv'

    diff_cols = [var for var in df.columns if 'DIFFERENCE' in var]
    results = []
    
    for var in diff_cols:
        stem = var.replace('DIFFERENCE_', '')

        var_change = dc.get_perturbation_change(stem)
        var_max = dc.get_perturbation_max(stem)
        var_min = dc.get_perturbation_min(stem)
        var_values = df[stem].values

        if var_change < 0:
            mask = np.where(var_values > var_min, True, False)
        elif var_change > 0:
            mask = np.where(var_values < var_max, True, False)

        counts = df[mask].groupby('labels')[var].count()

        results.append(counts)

    df = pd.concat(results, axis=1)

    df.to_csv(save_path / file_name, index=False)

    print(f'Saved {file_name} to {save_path}')

models = [
    '101_XGBoostClassifier.sav', '102_RFClassifier.sav', '103_LightGBMClassifier.sav'
]

if __name__ == '__main__':

    save_path = ROOT / 'results' / 'perturbation'
    X_test = pd.read_csv(ROOT / 'data' / 'perturbation' / 'X_test.csv', index_col=0)
    Y_test = pd.read_csv(ROOT / 'data' / 'perturbation' / 'Y_test.csv', index_col=0)
    test_set_clusters = pd.read_csv(ROOT / 'data' / 'perturbation' / 'test_set_clusters.csv', index_col=0)
    model_path = ROOT / 'models' / 'perturbation' / 'trained_models' / '103_LightGBMClassifier.sav'


    for col in ['Gend3', 'Disab2_POP', 'Eth7', 'WorkStat8', 'HHLiv9']:
        X_test[col] = X_test[col].astype(int).astype('category')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    scaler = model.get_scaler()

    target = 'active'
    group_by = 'labels'
    dc = DataCatalogue()
    to_perturb = dc.get_perturbation_vars()

    perturbations = create_perturbations(X_test, Y_test, model, scaler, to_perturb)

    perturbations[group_by] = test_set_clusters['LCA_Class'].values
    breakdowns = create_and_join_diff_series(perturbations, group_by)

    df_name = f'{model_path.stem}_perturbation_results_FINAL.csv'
    heatplot_name = f'{model_path.stem}_perturbation_heatplot_FINAL.png'

    perturbations.to_csv(save_path / df_name, index = False)
    print(f'Saved pertubation results to: {save_path / df_name}')

    create_save_heatplot(breakdowns, heatplot_name)

    print('Checking perturbation counts...')

    results_path = ROOT / 'results' /'perturbation' / df_name
    check_perturbed_counts(results_path)

    print(f'Finished perturbaiton process for {model.__class__.__name__}')

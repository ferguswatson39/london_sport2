import pandas as pd
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import seaborn as sns
import matplotlib.pyplot as plt
from src.loading_data.load_data import get_clean_2022
from src.perturbation.create_perturbations import build_perturbation_df
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
data = get_clean_2022()

def create_save_heatplot(breakdowns : pd.Series, target : str, heatplot_name : str):
    heatplot_path = ROOT / 'figures' / 'perturbation'
    if target == 'active':
        title = 'Average Change in the Probability of Participating in Over 150 Minutes of Moderate Activity Per Week'
    elif target == 'MEMS7_ALL':
        title = 'Average Change in the Minutes of Moderate Activity per week'
    heat = breakdowns.drop(columns = 'labels')
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        heat,
        annot=True,
        cmap="YlOrBr"
    )
    plt.title(title)
    plt.ylabel('Clusters')
    plt.savefig(heatplot_path / heatplot_name)
    print(f'Saved hatplot figure to: {heatplot_path / heatplot_name}')


def execute_perturbation_pipeline(df : pd.DataFrame, run_cases : dict):
    save_path = ROOT / 'results' / 'perturbation'

    for key, value in run_cases.items():
  
        model = run_cases[key]['model']
        target = run_cases[key]['target']
        group_by = 'labels'
        cluster_col = 'LCA_Class'
        df_name = f'{key}_{model.__class__.__name__}_perturbation_results.csv'
        heatplot_name = f'{model.__class__.__name__}_perturbation_heatplot.png'
        print(f'Starting Perturbation for {model.__class__.__name__}')
        perturbed_df, breakdowns = build_perturbation_df(df, model, target, group_by, cluster_col)
        perturbed_df.to_csv(save_path / df_name, index = False)
        print(f'Saved pertubation results to: {save_path / df_name}')
        create_save_heatplot(breakdowns, target, heatplot_name)
        print(f'Finished perturbaiton for {model.__class__.__name__}')






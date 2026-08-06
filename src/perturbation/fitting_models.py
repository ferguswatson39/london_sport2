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
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
from perturbation.models.classifiers.logistic import Logistic
from perturbation.models.classifiers.lgbm_classifier import LightGBMClassifier
from perturbation.models.classifiers.rf_classifier import RFClassifier
from perturbation.models.regressors.OLS_regressor import OLSRegressor
from perturbation.models.regressors.lgbm_regressor import LightGBMRegressor
from perturbation.models.regressors.rf_regressor import RFRegressor
from perturbation.models.classifiers.xgboost_classifier import XGBoostClassifier


classification_run_cases = {
    # XGBoost
    0 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : False, 'k_neighbors' : 0, 'strategy' : (0, 0)},

    1 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (500, 500)},
    2 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (500, 500)},
    3 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (500, 500)},
    4 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (500, 500)},

    #5 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (600, 600)},
    #6 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (600, 600)},
    #7 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (600, 600)},
    #8 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (600, 600)},

    #9 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (700, 700)},
    #10 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (700, 700)},
    #11 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (700, 700)},
    #12 : {'model' : XGBoostClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (700, 700)},

    # Random Forest
    13 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : False, 'k_neighbors' : 0, 'strategy' : (0, 0)},

    14 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (500, 500)},
    15 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (500, 500)},
    16 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (500, 500)},
    17 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (500, 500)},

    #18 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (600, 600)},
    #19 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (600, 600)},
    #20 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (600, 600)},
    #21 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (600, 600)},

    #22 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (700, 700)},
    #23 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (700, 700)},
    #24 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (700, 700)},
    #25 : {'model' : RFClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (700, 700)},


    # LightGBM 
    26 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : False, 'k_neighbors' : 0, 'strategy' : (0, 0)},

    27 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (500, 500)},
    28 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (500, 500)},
    29 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (500, 500)},
    30 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (500, 500)},

    #31 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (600, 600)},
    #32 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (600, 600)},
    #33 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (600, 600)},
    #34 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (600, 600)},

    #35 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 2, 'strategy' : (700, 700)},
    #36 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 3, 'strategy' : (700, 700)},
    #37 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 4, 'strategy' : (700, 700)},
    #38 : {'model' : LightGBMClassifier(),'target' : 'active_status', 'smote' : True, 'k_neighbors' : 5, 'strategy' : (700, 700)},


}

def preprocess_perturbation(df : pd.DataFrame, cluster_column : str, target : str, to_encode_vars : list[str]):

    # Will remove this preprocessing when merge with master (06/08)
    df['MEMS7_ALL'] = df['MEMS7_ALL'].clip(upper=6720)
    active_groupings = [df['MEMS7_ALL'] == 0, (df['MEMS7_ALL'] > 0) & (df['MEMS7_ALL'] < 150), df['MEMS7_ALL'] >= 150]
    values = [0.0, 1.0, 2.0]
    df['active_status'] = np.select(active_groupings, values)

    df['MEMS7_ALL'] = np.log1p(df['MEMS7_ALL'])
    df['Disab2_POP'] = df['Disab2_POP'].replace({1.0 : 0.0, 2.0 : 1.0})
    

    assert to_encode_vars == ['Gend3', 'Eth7', 'WorkStat8', 'HHLiv9']

    df_encoded = one_hot_encode_frame(df, to_encode_vars, None)

    drop_cols = ['serial', 'year', 'LCA_Class', 'MEMS7_ALL', 'active', 'active_status']
    keep_cols = [col for col in df_encoded.columns if col not in drop_cols]
    print(f'Keep columns : {keep_cols}')

    labels = df_encoded[cluster_column]
    Y = df_encoded[target]
    X = df_encoded[keep_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels)
    test_set_clusters = df_encoded.loc[X_test.index, cluster_column]

    print(f'X_train cols:\n{X_train.columns}')
    print(f'X_train shape:\n{X_train.shape}')
    print(f'X_train values:\n{X_train.head(10)}')

    print(f'X_test cols:\n{X_test.columns}')
    print(f'X_test shape:\n{X_test.shape}')
    print(f'X_test values:\n{X_test.head(10)}')

    print(f'Y_train value counts:\n {Y_train.value_counts()}')
    print(f'Y_train shape:\n{Y_train.shape}')
    print(f'Y_train values:\n{Y_train.head(10)}')

    print(f'Y_test value counts:\n {Y_test.value_counts()}')
    print(f'Y_test shape:\n{Y_test.shape}')
    print(f'Y_test values:\n{Y_test.head(10)}')

    save_path = ROOT / 'data' / 'perturbation'

    X_test.to_csv(save_path / 'X_test.csv')
    Y_test.to_csv(save_path / 'Y_test.csv')
    test_set_clusters.to_csv(save_path / 'test_set_clusters.csv')

    print(f'Saved X_test, Y_test, test_set_clusters to {save_path}')
    
    return X_train, X_test, Y_train, Y_test, test_set_clusters

def smote_resample(X_train : pd.DataFrame, Y_train : pd.Series, k_neighbors : int, strategy : tuple):
    smote = SMOTEN(
        k_neighbors= k_neighbors, 
        sampling_strategy= {0 : strategy[0], 1: strategy[1]}, 
        random_state = 42
    )
    X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

    print(f'Pre smote: {X_train.shape}')
    print(f'Post smote: {X_train_smote.shape}')

    print(f'Pre smote: {Y_train.value_counts()}')
    print(f'Post smote: {Y_train_smote.value_counts()}')

    return X_train_smote, Y_train_smote



if __name__ == '__main__':
    # Can edit run cases here to change run cases
    run_cases = classification_run_cases

    target = 'active_status'
    cluster_col = 'LCA_Class'

    dc = DataCatalogue()
    df = get_clean_2022()
    df_copy = df.copy()

    to_perturb = dc.get_perturbation_vars()
    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()
    to_encode_vars = dc.get_perturbation_core_to_encode()


    X_train_main, X_test, Y_train_main, Y_test, test_set_clusters = preprocess_perturbation(df_copy, cluster_col, target, to_encode_vars)

    for key, value in run_cases.items():

        X_train , Y_train = X_train_main, Y_train_main

        run_num = key
        model = run_cases[key]['model']
        do_smote = run_cases[key]['smote']
        k_neigh = run_cases[key]['k_neighbors']
        strat = run_cases[key]['strategy']

        if do_smote:
            print(f'Resampling with smote for run number: {run_num}')
            X_train, Y_train = smote_resample(X_train, Y_train, k_neigh, strat)
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Added scaler object so that it can be used to transform future X_test inputs
        model.fit(X_train_scaled, Y_train, scaler)

        model.get_preds(X_test_scaled, Y_test, save_metric = True)

        model.save_class(run_num)




"""

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

classification_cases = {
    0 : {'model' : XGBoostClassifier(), 'target' : 'active_status'},
    1 : {'model' : RFClassifier(),'target' : 'active_status'},
    2 : {'model' : LightGBMClassifier(),'target' : 'active_status'},
}

"""
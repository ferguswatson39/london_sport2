import pandas as pd
import sys
import pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.data_catalogue import DataCatalogue
from src.loading_data.load_data import get_clean_2022
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perturbation.models.lgbm_classifier import LightGBMClassifier
from perturbation.models.rf_classifier import RFClassifier
from perturbation.models.xgboost_classifier import XGBoostClassifier

"""
File defines the classifier fitting pipeline across XGBOOST, LIGHTGBM and RANDOMFOREST models.

Conducts model comparison at the end of the pipeline to determine which model had the best results.
"""

run_cases = {
    101 : {'model' : XGBoostClassifier(),'target' : 'active'},
    102 : {'model' : RFClassifier(),'target' : 'active'},
    103 : {'model' : LightGBMClassifier(),'target' : 'active'}
}

def preprocess_perturbation(df : pd.DataFrame, cluster_column : str, target : str, categoricals : list[str], save = bool):

    assert df.columns.to_list() == [
        'LCA_Class', 'Age9', 'Gend3', 'Eth7', 'Disab2_POP', 'Educ6', 'NSSEC5',
        'IMD10', 'WorkStat8', 'Child4', 'HHLiv9', 'active', 'Motiva_POP',
        'motivd_POP', 'inclus_a', 'inclus_b', 'inclus_c', 'anxious', 'comm1',
        'comm2', 'happy', 'indev', 'indevtry', 'lone', 'worthw'
       ]

    assert categoricals == ['active', 'Gend3', 'Disab2_POP', 'Eth7', 'WorkStat8', 'HHLiv9']

    for col in categoricals:
        df[col] = df[col].astype(int).astype('category')


    drop_cols = ['LCA_Class', 'active']
    keep_cols = [col for col in df.columns if col not in drop_cols]
    print(f'Keep columns : {keep_cols}')

    
    assert keep_cols == ['Age9', 'Gend3', 'Eth7', 'Disab2_POP', 'Educ6', 'NSSEC5',
                        'IMD10', 'WorkStat8', 'Child4', 'HHLiv9', 'Motiva_POP',
                        'motivd_POP', 'inclus_a', 'inclus_b', 'inclus_c', 'anxious',
                        'comm1', 'comm2', 'happy', 'indev', 'indevtry', 'lone', 'worthw']

    labels = df[cluster_column]
    Y = df[target]
    X = df[keep_cols]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 42, stratify = labels, shuffle = True)
    test_set_clusters = df.loc[X_test.index, cluster_column]

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

    if save:

        save_path = ROOT / 'data' / 'perturbation'

        X_test.to_csv(save_path / 'X_test.csv')
        Y_test.to_csv(save_path / 'Y_test.csv')
        test_set_clusters.to_csv(save_path / 'test_set_clusters.csv')

        print(f'Saved X_test, Y_test, test_set_clusters to {save_path}')
    
    return X_train, X_test, Y_train, Y_test, test_set_clusters

if __name__ == '__main__':
    # Can edit run cases here to change run cases
    run_cases = run_cases

    target = 'active'
    cluster_col = 'LCA_Class'

    dc = DataCatalogue()
    df = get_clean_2022()

    to_perturb = dc.get_perturbation_vars()
    continuous_vars = dc.get_perturbation_core_contins() + dc.get_perturbation_vars()
    categoricals = dc.get_perturbation_processing_categoricals()

    X_train_main, X_test_main, Y_train_main, Y_test_main, test_set_clusters = preprocess_perturbation(df, cluster_col, target, categoricals, save = True)

    for key, value in run_cases.items():

        X_train, X_test, Y_train, Y_test = X_train_main.copy(), X_test_main.copy(), Y_train_main.copy(), Y_test_main.copy()

        run_num = key
        model = run_cases[key]['model']

        scaler = StandardScaler()
        X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars])
        # X_test is saved as a csv during preprocess perturbations so the scaling here doesnt matter
        X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])

        test_categoricals = [col for col in categoricals if col != 'active']
        if (X_test.dtypes[test_categoricals ] != 'category').any():
            raise ValueError('Catgegorical columns do not have categorical dtype.')
        
        # Added scaler object so that it can be used to transform future X_test inputs
        model.fit(X_train, Y_train, scaler)

        model.get_preds(X_test, Y_test, save_metric = True)

        model.save_class(run_num)

    print('Finished fitting models')
    print('Verifying model metrics...')

    model_folder = ROOT / 'models' / 'perturbation' / 'trained_models'
    file_paths = [(file, file.name) for file in model_folder.iterdir()]
    best_f1 = float('-inf')
    best_model = None
    for path, file_name in file_paths:
        with open(path, 'rb') as file:
            model = pickle.load(file)
            if best_f1 < model.get_f1():
                best_f1 = model.get_f1()
                best_model = file_name
            print('________________________________')
            print(file_name)
            print(model.get_confusion())
            print(f'F1 Score: {model.get_f1()}')
            print(f'ROC AUC Score: {model.get_roc_auc()}')
            print(f'Accuracy: {model.get_accuracy()}')
            print(model.hyperparams)
            print(model.get_model().get_params())
    print('__________________________')
    print('__________________________')
    print(f'BEST Model:\n>>>> Name: {best_model}\n>>>> F1: {best_f1}')
from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_data
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.loading_data.data_catalogue import DataCatalogue
from src.stream_2.check_empty_dummies import check_empty_dummies
import matplotlib.pyplot as plt
import seaborn as sns

def check_multicolinearity() -> pd.DataFrame:
    dc = DataCatalogue()
    df = get_data()
    discrete_variables = dc.get_to_be_encoded_vars()
    # reference_categories = [f'{c}_{df[c].value_counts().index[0]}' for c in discrete_variables]
    encoder = OneHotEncoder(drop='first', handle_unknown = 'ignore', sparse_output = False).set_output(transform = 'pandas')
    encoded = encoder.fit_transform(df[discrete_variables])
    df_encoded = pd.concat([df, encoded], axis = 1).drop(columns = discrete_variables)
    all_years = []
    for year in sorted(df_encoded['year'].unique()):
        df_encoded_sy = df_encoded[df_encoded['year'] == year]
        df_encoded_sy = df_encoded_sy.drop(columns=['year'])
        Y = df_encoded_sy['LOG_MEMS7_ALL']
        X = df_encoded_sy.drop(columns=dc.get_target_vars())
        corr = X.corr()
        return corr
    
corr = check_multicolinearity()
# sns.heatmap(corr)
# plt.show()
print(type(corr))
corr = corr[corr.abs() > 0.65]
sns.heatmap(corr)
plt.show()
# corr.to_csv('corr.csv')
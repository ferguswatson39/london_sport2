from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'src' / 'stream_1'))
sys.path.append(str(ROOT / 'src' / 'stream_3' / 'latent_class_analysis'))

import numpy as np
import pandas as pd

from bounded_regression import LogisticBoroughForecaster
from bayesian_regression import BayesianBoroughForecaster
from constants_lca import class_names as CLASS_LABELS

import lca_bayesian_ridge
import lca_bounded_regression
from lca_bayesian_ridge import bayesian_ridge_forecast_classes
from lca_bounded_regression import bounded_logistic_forecast_classes


TRAIN_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
FORECAST_YEARS = [2024, 2025, 2026, 2027]
ADJUST_COVID = False

for module in (lca_bayesian_ridge, lca_bounded_regression):
    module.included_years = TRAIN_YEARS
    module.forecast_years = FORECAST_YEARS

OUT_DIR = ROOT / 'visualisation' / 'data'
CLUSTER_DATA = ROOT / 'data' / 'master_data' / 'latent_class_forecasting_data.csv'

METRIC_COLUMNS = ['mems_mins_per_week', 'pct_active', 'pct_volunteering']


def build_boroughs():
    columns = {}

    mems = BayesianBoroughForecaster(
        INCLUDED_YEARS=TRAIN_YEARS,
        FORECAST_YEARS=FORECAST_YEARS,
        target_col='MEMS7_ALL',
        ADJUST_COVID=ADJUST_COVID,
    ).fit_predict()
    columns['mems_mins_per_week'] = mems.set_index(['borough', 'year'])['target']

    for name, target_col in [('pct_active', 'active'),
                             ('pct_volunteering', 'VolAny')]:
        df = LogisticBoroughForecaster(
            INCLUDED_YEARS=TRAIN_YEARS,
            FORECAST_YEARS=FORECAST_YEARS,
            target_col=target_col,
            ADJUST_COVID=ADJUST_COVID,
        ).fit_predict()
        columns[name] = df.set_index(['borough', 'year'])['target']

    out = pd.DataFrame(columns).reset_index()
    return out.sort_values(['borough', 'year']).reset_index(drop=True)


def build_clusters():
    raw = pd.read_csv(CLUSTER_DATA)
    raw['calendar_year'] = raw['calendar_year'].astype(int)
    raw['LCA_Class'] = raw['LCA_Class'].astype(int)

    sources = [
        ('mems_mins_per_week', 'Mean_MEMS7_ALL', bayesian_ridge_forecast_classes),
        ('pct_active',         'Mean_active',    bounded_logistic_forecast_classes),
        ('pct_volunteering',   'Mean_VolAny',    bounded_logistic_forecast_classes),
    ]

    columns = {}
    for name, source_col, forecast_fn in sources:
        df = forecast_fn(raw, source_col)
        df['LCA_Class'] = df['LCA_Class'].astype(int)
        columns[name] = (df.rename(columns={'calendar_year': 'year'})
                           .set_index(['LCA_Class', 'year'])['value'])

    out = pd.DataFrame(columns).reset_index()
    out['cluster'] = out['LCA_Class'].map(CLASS_LABELS)
    out = out.rename(columns={'LCA_Class': 'lca_class'})
    out = out[['lca_class', 'cluster', 'year'] + METRIC_COLUMNS]
    return out.sort_values(['cluster', 'year']).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in [('borough', build_boroughs()), ('cluster', build_clusters())]:
        df[METRIC_COLUMNS] = df[METRIC_COLUMNS].round(2)
        df['period'] = np.where(df['year'] >= FORECAST_YEARS[0], 'forecast', 'historical')
        path = OUT_DIR / f'tableau_{name}_forecasts.csv'
        df.to_csv(path, index=False)
        print(f'{name}: {len(df):,} rows, {df[METRIC_COLUMNS].isna().sum().sum()} NaN '
              f'-> {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
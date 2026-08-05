from constants_lca import (included_years, forecast_years)
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np

def bayesian_ridge_forecast_classes(forecasting_df, target_col):
    output = []
    X_train = (np.array(included_years) - included_years[0]).reshape(-1, 1)
    X_forecast = (np.array(forecast_years) - included_years[0]).reshape(-1, 1)

    for cls in sorted(forecasting_df["LCA_Class"].unique()):

        tmp = (forecasting_df[(forecasting_df["LCA_Class"] == cls) & (forecasting_df["calendar_year"].isin(included_years))].sort_values("calendar_year"))

        if len(tmp) != len(included_years):
            print(f"Skipping Class {cls}: missing years.")
            continue

        y_train = tmp[target_col].values

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, max_iter=300, tol=1e-3)

        model.fit(X_train, y_train)
        r2 = model.score(X_train, y_train)

        y_forecast, forecast_std = model.predict(X_forecast, return_std=True)

        years_full = included_years + forecast_years
        values_full = np.concatenate((y_train, y_forecast))
        error_full = np.concatenate((np.zeros(len(y_train)), forecast_std))

        for year, value, error in zip(years_full, values_full, error_full):
            output.append({"LCA_Class": cls, "calendar_year": year, "value": value, "error": error, "R2": r2})

    return pd.DataFrame(output)
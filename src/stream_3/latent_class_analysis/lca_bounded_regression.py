from constants_lca import (included_years, forecast_years)
from sklearn.linear_model import LinearRegression
from scipy.special import logit, expit
import pandas as pd
import numpy as np

def bounded_logistic_forecast_classes(forecasting_df, target_col):
    output = []
    X_train = (np.array(included_years) - included_years[0]).reshape(-1, 1)
    X_forecast = (np.array(forecast_years) - included_years[0]).reshape(-1, 1)

    for cls in sorted(forecasting_df["LCA_Class"].unique()):

        tmp = (forecasting_df[(forecasting_df["LCA_Class"] == cls) & (forecasting_df["calendar_year"].isin(included_years))].sort_values("calendar_year"))

        if len(tmp) != len(included_years):
            print(f"Skipping Class {cls}: missing years.")
            continue

        y_train = tmp[target_col].values

        buffer = max(0.05, (y_train.max() - y_train.min()) / 5)
        lower = max(0, y_train.min() - buffer)
        upper = min(1.0, y_train.max() + buffer)
        y_scaled = (y_train - lower) / (upper - lower)

        model = LinearRegression()
        model.fit(X_train, logit(y_scaled))
        r2 = model.score(X_train, logit(y_scaled))

        y_forecast = (lower + expit(model.predict(X_forecast)) * (upper - lower))

        years_full = included_years + forecast_years
        values_full = np.concatenate((y_train, y_forecast)) * 100

        for year, value in zip(years_full, values_full):
            output.append({"LCA_Class": cls, "calendar_year": year, "value": value, "R2": r2})

    return pd.DataFrame(output)
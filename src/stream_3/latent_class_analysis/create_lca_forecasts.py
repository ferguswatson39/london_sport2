from lca_bayesian_ridge import bayesian_ridge_forecast_classes
from forecast_visualisation import plot_forecasts
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

target_col = "Mean_MEMS7_ALL"

forecasting_summary_path = (ROOT/"data"/"master_data"/"latent_class_forecasting_data.csv")

forecast_path = (ROOT/"exploration"/"forecasting_eda"/f"lca_{target_col.lower()}_bayesian_forecast.csv")

plot_path = (ROOT/"exploration"/"forecasting_eda"/f"lca_{target_col.lower()}_bayesian_forecast.png")

forecasting_summary = pd.read_csv(forecasting_summary_path)

forecasting_summary["calendar_year"] = forecasting_summary["calendar_year"].astype(int)

forecast_df = bayesian_ridge_forecast_classes(forecasting_summary, target_col)

# forecast_df.to_csv(forecast_path, index=False)

plot_forecasts(forecast_df, target_col, save_path = None)
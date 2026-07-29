from lca_postprocessing import (create_class_summary, assign_classes, create_master_dataset, create_forecasting_summary)
from lca_visualisation import (calculate_dominance, plot_heatmap, plot_mems_trends)
from latent_class_analysis import load_model
from constants_lca import cluster_cols
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

overall_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_full_preprocessed_data_set.csv.gz")

cluster_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_clustering_data_set.csv")

class_summary_path = (ROOT/"exploration"/"clustering"/"stepmix_25_class_summary.csv")

master_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_master_clustering_data_set.csv")

forecasting_path = (ROOT/"exploration"/"data"/"master_data"/"latent_class_forecasting_data_set.csv")

model_path = (ROOT/"models"/"stream_3"/"stepmix_new_25.pkl")

overall_df = pd.read_csv(overall_path)

cluster_df = pd.read_csv(cluster_path)

lca = load_model(model_path)

model = lca.model
measurement = model.get_parameters()["measurement"]
pis = measurement["pis"]
block_size = measurement["max_n_outcomes"]
class_probabilities = pis.reshape(pis.shape[0], len(cluster_cols), block_size)

class_summary, class_sizes = create_class_summary(lca, cluster_df)

cluster_df = assign_classes(lca, cluster_df)

master_df = create_master_dataset(overall_df, cluster_df)

forecasting_summary = create_forecasting_summary(master_df)

class_summary.to_csv(class_summary_path, index=False)

master_df.to_csv(master_path, index=False)

forecasting_summary.to_csv(forecasting_path, index=False)

dominance = calculate_dominance(class_probabilities, class_sizes, cluster_df)

# print(dominance.round(2))

plot_heatmap(class_probabilities, cluster_df)

plot_mems_trends(forecasting_summary, metric = "Mean_MEMS7_ALL")
from clustering_preprocessing import prepare_clustering_dataset
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

data_path = (ROOT/"data"/"master_data"/"2016_to_2023_full_preprocessed_data_set.csv.gz")

output_path = (ROOT/"data"/"master_data"/"2016_to_2023_clustering_input_data.csv")

master_df = pd.read_csv(data_path)

cluster_df = prepare_clustering_dataset(master_df)

# cluster_df.to_csv(output_path, index=False)
from clustering_preprocessing import (create_motivation_pc, prepare_clustering_dataset)
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

data_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_full_preprocessed_data_set.csv.gz")

output_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_clustering_data_set.csv")

master_df = pd.read_csv(data_path)

master_df, loadings = create_motivation_pc(master_df)

cluster_df = prepare_clustering_dataset(master_df)

cluster_df.to_csv(output_path, index=False)
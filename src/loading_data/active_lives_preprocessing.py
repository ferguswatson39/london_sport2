from active_lives_loading import (load_all_surveys, align_surveys, create_dictionary, report_missingness)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

output_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_full_preprocessed_data_set.csv.gz")

datasets, meta = load_all_surveys()

dictionary = create_dictionary(meta)

master_df = align_surveys(datasets)

master_df = master_df[master_df["LondInOut"].notna()]

missing_summary = report_missingness(master_df)

# master_df.to_csv(output_path, index = False, compression = "gzip")
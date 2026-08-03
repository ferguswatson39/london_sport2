from active_lives_loading import (load_all_surveys, align_surveys, create_dictionary, report_missingness)
from correct_calendar import calendar_adjustment
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

output_path = (ROOT/"data"/"master_data"/"2016_to_2023_full_preprocessed_data_set.csv.gz")

datasets, meta = load_all_surveys()

dictionary = create_dictionary(meta)

master_df = align_surveys(datasets)

master_df = master_df[master_df["LondInOut"].notna()]

master_df = calendar_adjustment(master_df)

master_df['MEMS7_ALL'] = master_df['MEMS7_ALL'].clip(upper = 6720)

master_df['active'] = master_df['MEMS7_ALL'] >= 150

missing_summary = report_missingness(master_df)

# master_df.to_csv(output_path, index = False, compression = "gzip")
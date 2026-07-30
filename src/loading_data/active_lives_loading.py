from constants_preprocessing import (drop_prefs, rename_map, data_files)
from collections import Counter
import pandas as pd
import pyreadstat
import re

def drop_prefix(df, prefixes):
    cols = [c for c in df.columns if c.startswith(prefixes)]
    df.drop(columns=cols, inplace=True)

def clean_dataset(df):
    df.rename(columns=rename_map, inplace=True)
    drop_prefix(df, drop_prefs)
    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > 0.95]
    cols_to_drop = high_missing.index.difference(["LondInOut"])
    df.drop(columns=cols_to_drop, inplace=True)
    drop_cols = []
    for col in df.columns:
            if re.search(r'_([D-EG-U])\d+$', col):
                drop_cols.append(col)
    df.drop(columns=drop_cols, inplace=True)
    return df

def load_survey(year, path):
     df, meta = pyreadstat.read_sav(path)
     df["year"] = year
     df = clean_dataset(df)
     return df, meta

def load_all_surveys():
    datasets = []
    for year, filename in data_files.items():
        df, meta = load_survey(year, filename)
        datasets.append(df)
    return datasets, meta

def align_surveys(datasets):
    latest_df = datasets[-1]
    latest_cols = list(latest_df.columns)

    var_counts = Counter()
    for df in datasets:
        var_counts.update(df.columns)

    keep_cols = []
    for col in latest_cols:
        if var_counts[col] >= 4:
            keep_cols.append(col)

    aligned = []
    for df in datasets:
        df_copy = df.reindex(columns = keep_cols)
        aligned.append(df_copy)

    return pd.concat(aligned, ignore_index=True)

def create_dictionary(meta):
    return pd.DataFrame({"variable": meta.column_names, "label": meta.column_labels})

def report_missingness(df):
    missing_summary = (df.isna().mean().sort_values(ascending=False).reset_index())
    missing_summary.columns = ["variable", "missing_pct"]
    print(missing_summary.head(50))
    return missing_summary
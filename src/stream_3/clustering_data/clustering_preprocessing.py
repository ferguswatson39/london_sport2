from constants_clustering import (workstat_mapping, hhliv_mapping, cluster_cols)

def recode_nssec(df):
    df = df.copy()
    df.loc[df["NSSEC5"].isna() & (df["Age9"] >= 8), "NSSEC5"] = 5
    return df

def recode_workstat(df):
    df = df.copy()
    df["WorkStat8"] = df["WorkStat10"].map(workstat_mapping)
    return df

def recode_hhliv(df):
    df = df.copy()
    df["HHLiv9"] = df["HHLiv12"].map(hhliv_mapping)
    return df

def prepare_clustering_dataset(df):
    df = recode_nssec(df)
    df = recode_workstat(df)
    df = recode_hhliv(df)
    cluster_df = df[cluster_cols].copy()
    cluster_df = cluster_df.dropna().copy()
    cluster_vars = [col for col in cluster_cols if col not in ["serial", "year"]]
    cluster_df[cluster_vars] = cluster_df[cluster_vars].astype(int)
    for col in cluster_vars:
        cluster_df[col] -= cluster_df[col].min()
    return cluster_df


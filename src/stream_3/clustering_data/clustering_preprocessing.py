from constants_clustering import (motivation_cols, workstat_mapping, hhliv_mapping, cluster_cols)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

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

def create_motivation_pc(df):
    motivation_df = df[motivation_cols].dropna().copy()
    scaler = StandardScaler()
    motivation_scaled = scaler.fit_transform(motivation_df)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(motivation_scaled).flatten()
    motivation_df["Motivation_PC"] = pc1
    df = df.merge(motivation_df[["Motivation_PC"]], left_index=True, right_index=True, how="left")
    df["Motivation_PC_Q"] = pd.qcut(df["Motivation_PC"], q=5, labels=False)
    loadings = pd.DataFrame(pca.components_.T, index=motivation_cols, columns=["Loading"])
    return df, loadings

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


from constants_lca import (cluster_cols, value_labels, master_columns, forecast_aggregations)
import pandas as pd
import numpy as np

def create_class_summary(lca, cluster_df):

    model = lca.model

    measurement = model.get_parameters()["measurement"]
    pis = measurement["pis"]
    block_size = measurement["max_n_outcomes"]

    n_classes = pis.shape[0]
    original_codes = {col: np.sort(cluster_df[col].dropna().unique()) for col in cluster_cols}
    n_categories = [cluster_df[col].nunique() for col in cluster_cols]
    X = cluster_df[cluster_cols]

    posterior = lca.predict_proba(X)
    class_sizes = posterior.mean(axis=0)    

    summary_rows = []

    for cls in range(n_classes):
        row = {"Class": cls, "Size": f"{class_sizes[cls]:.1%}"}

        for i, (var, ncat) in enumerate(zip(cluster_cols, n_categories)):
            start = i * block_size
            probs = pis[cls, start:start + block_size][:ncat]
            top = np.argsort(probs)[::-1][:3]
            entries = []
            label_dict = value_labels[var]

            for idx in top:
                code = original_codes[var][idx]
                label = label_dict[code]
                entries.append(f"{label} ({probs[idx]:.0%})")

            row[var] = ", ".join(entries)

        summary_rows.append(row)

    class_summary = pd.DataFrame(summary_rows)

    return class_summary, class_sizes

def assign_classes(lca, cluster_df):

    X = cluster_df[cluster_cols]
    cluster_df = cluster_df.copy()
    cluster_df["Class"] = lca.predict(X)

    return cluster_df

def create_master_dataset(overall_df, cluster_df):

    assert cluster_df.set_index(["year", "serial"]).index.is_unique

    merge_cols = [col for col in overall_df.columns if col not in cluster_cols]
    master_df = cluster_df.merge(overall_df[["year", "serial"] + merge_cols], on=["year", "serial"], how="left", validate="one_to_one")
    master_df = master_df.rename(columns={"Class": "LCA_Class"})
    master_df = master_df[master_columns]
    master_df["LCA_Class"] = master_df["LCA_Class"].astype("Int64")

    return master_df

def create_forecasting_summary(master_df):

    forecasting_summary = (master_df.groupby(["LCA_Class", "year"]).agg(**forecast_aggregations).reset_index())
    
    return forecasting_summary
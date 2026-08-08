from constants_lca import (cluster_cols, value_labels, display_names, class_names)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def calculate_dominance(class_probabilities, class_sizes, cluster_df):

    n_classes = class_probabilities.shape[0]
    n_categories = [cluster_df[col].nunique() for col in cluster_cols]
    dominance_rows = []

    for cls in range(n_classes):
        row = {"Class": cls, "Size": round(class_sizes[cls], 3)}

        for i, (var, ncat) in enumerate(zip(cluster_cols, n_categories)):
            probs = class_probabilities[cls, i, :ncat]
            row[var] = probs.max()

        dominance_rows.append(row)

    dominance = pd.DataFrame(dominance_rows)

    return dominance

def plot_heatmap(class_probabilities, cluster_df):

    n_classes = class_probabilities.shape[0]
    n_plots = len(cluster_cols)
    ncols = 4
    nrows = math.ceil(n_plots / ncols)

    original_codes = {col: np.sort(cluster_df[col].dropna().unique()) for col in cluster_cols}
    n_categories = [cluster_df[col].nunique() for col in cluster_cols]

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    for i, (ax, var) in enumerate(zip(axes, cluster_cols)):
        ncat = n_categories[i]
        population = (cluster_df[var].value_counts(normalize=True).sort_index().values)
        heat = class_probabilities[:, i, :ncat] - population
        codes = original_codes[var]
        labels = [value_labels[var][code] for code in codes]

        if var == "IMD10":
            labels = [f"D{i+1}" for i in range(len(labels))]

        sns.heatmap(heat,
                    ax=ax,
                    cmap="RdBu_r",
                    center=0,
                    vmin=-0.5,
                    vmax=0.5,
                    linewidths=0.3,
                    linecolor="white",
                    xticklabels=labels,
                    yticklabels=False)

        ax.set_title(display_names[var], fontsize=11, fontweight="bold")

        if i % ncols == 0:
            ax.set_yticks(np.arange(n_classes) + 0.5)
            ax.set_yticklabels(range(1, n_classes + 1), fontsize=8)
            ax.set_ylabel("Class")
        else:
            ax.set_yticks([])

        ax.tick_params(axis="x", labelsize=9, rotation=30)

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.suptitle("Latent class profiles relative to the London population", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_mems_trends(forecasting_summary, metric = "Median_MEMS7_ALL"):

    n_classes = forecasting_summary["LCA_Class"].nunique()
    ncols = 5
    nrows = math.ceil(n_classes / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    
    years = sorted(forecasting_summary["calendar_year"].unique())
    ymax = forecasting_summary[metric].max()

    for cls in range(n_classes):
            ax = axes[cls]
            tmp = forecasting_summary[forecasting_summary["LCA_Class"] == cls]
            tmp = tmp.sort_values("calendar_year")
            ax.plot(tmp["calendar_year"], tmp[metric], marker="o", linewidth=2)
            ax.set_xticks(years)
            ax.set_title(class_names.get(cls, f"Class {cls + 1}"), fontsize=9, fontweight="bold")
            ax.grid(alpha=0.3)
            padding = 0.05 * ymax
            ax.set_ylim(0, ymax + padding)

    for ax in axes[n_classes:]:
                    ax.axis("off")

    for ax in axes[-ncols:]:
            ax.set_xlabel("Year", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

    for ax in axes[::ncols]:
            ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=9)

    fig.suptitle(f"{metric.replace('_', ' ').title()} over time by latent class", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

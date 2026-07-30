from constants_lca import (included_years, forecast_years, class_names)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecasts(forecast_df, target_col, save_path = None):
    plot_df = forecast_df.copy()
    plot_df["ClassName"] = plot_df["Class"].map(class_names)
    plot_df["Forecast"] = plot_df["year"].isin(forecast_years)
    colours = (["#808080"] * len(included_years) + ["#00BFFF"] * len(forecast_years))

    g = sns.relplot(kind="scatter",
                    data=plot_df,
                    x="year",
                    y="value",
                    col="ClassName",
                    col_wrap=5,
                    height=2.3,
                    aspect=1.4,
                    s=50,
                    palette="RdYlGn",
                    hue="value",
                    legend=False,
                    edgecolors=colours,
                    linewidth=1,
                    zorder=2)

    for cls_name, ax in g.axes_dict.items():

        class_df = (plot_df[plot_df["ClassName"] == cls_name].sort_values("year"))

        sns.regplot(data=class_df,
                    x="year",
                    y="value",
                    scatter=False,
                    ci=None,
                    color="#00BFFF",
                    line_kws={"alpha":0.4, "zorder":0},
                    ax=ax)

        future = class_df[class_df["year"].isin(forecast_years)]

        ax.errorbar(x=future["year"],
                    y=future["value"],
                    yerr=1.96*future["error"],
                    fmt="none",
                    color="#00BFFF",
                    alpha=0.2,
                    capsize=3,
                    zorder=1)

        ax.axvline(included_years[-1], linestyle=":", color="black", alpha=0.5)

    g.set_titles("{col_name}", weight="bold", size=9)
    g.set(xlabel="Year", ylabel=target_col.replace("_", " ").title())
    g.set_xticklabels([])
    g.set_yticklabels([])
    
    plt.subplots_adjust(top=0.94)

    g.figure.suptitle(f"Bayesian Ridge Forecasts of {target_col.replace('_', ' ').title()} by Latent Class", fontsize=16, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
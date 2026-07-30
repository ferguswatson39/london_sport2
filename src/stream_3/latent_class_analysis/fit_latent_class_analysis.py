from latent_class_analysis import LatentClassAnalysis
from constants_lca import cluster_cols
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

data_path = (ROOT/"exploration"/"data"/"master_data"/"2016_to_2023_clustering_data_set.csv")

output_path = (ROOT/"exploration"/"data"/"master_data"/"stepmix_latent_class_analysis_results.csv")

cluster_df = pd.read_csv(data_path)

X = cluster_df[cluster_cols]

results = []

for k in range(12, 31):

    print(f"Fitting {k} classes...")

    try:
        lca = LatentClassAnalysis(n_components=k, n_init=10, max_iter=1000, abs_tol=1e-5)
        lca.fit(X)
        results.append(lca.evaluate(X))

    except Exception as e:
        results.append({"Classes": k, "Error": str(e)})

    # pd.DataFrame(results).to_csv(output_path, index=False)

results_df = pd.DataFrame(results)

print(results_df)
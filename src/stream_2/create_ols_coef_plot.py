from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt

save_path = Path(ROOT / 'figures' / 'ols_plot.png')
data_path = Path(ROOT / 'data' / 'ols_coefficients' / 'ols_coef_results.csv')
df = pd.read_csv(data_path)
df = df[df['feature_names'] != 'const']
# cols = [c for c in df.columns if c != 'coef_as_percent']
pivoted = df.pivot(
    index='feature_names',
    columns='year',
    values=['coef_as_percent', 'pvalues', 'confidence_lower', 'confidence_upper', 'std_error']
)
fig, ax = plt.subplots(figsize=(10,15))
num_years = len(pivoted.columns.levels[1])
ax.set_prop_cycle(color = plt.cm.YlOrRd(np.linspace(0,1,num_years)))
for idx, year in enumerate(sorted(pivoted.columns.levels[1])):
    ax.errorbar(x=pivoted['coef_as_percent'][year].values, y =pivoted.index, marker='o', linestyle='None', label=year)
    ax.legend()
    ax.axvline(x=0, linestyle='--')
    ax.set_title('Percentage change in MEMS, for a one unit change in the predictor, 2016-2023')
fig.savefig(save_path)
print(f'Done. Figure saved to {save_path}')


from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt

save_path = Path(ROOT / 'figures' / 'logistic_plot.png')
data_path = Path(ROOT / 'data' / 'logistic_coefficients' / 'logistic_coef_results.csv')
df = pd.read_csv(data_path)
cols = [c for c in df.columns if c != 'odds_ratios']
pivoted = df.pivot(
    index='feature_names',
    columns='year',
    values=['odds_ratios', 'pvalues', 'confidence_lower', 'confidence_upper', 'std_error']
)
fig, ax = plt.subplots(figsize=(10,15))
ax.set_prop_cycle(color = plt.cm.YlOrRd(np.linspace(0,1,8)))
for idx, year in enumerate(sorted(pivoted.columns.levels[1])):
    ax.errorbar(x=pivoted['odds_ratios'][year].values, y =pivoted.index, marker='o', linestyle='None', label=year)
    ax.legend()
    ax.set_title('Odds ratio of participating in >150 minutes of PA per week, 2016-2023')
    ax.axvline(x=1, linestyle='--')
fig.savefig(save_path)
print(f'Done. Figure saved to {save_path}')


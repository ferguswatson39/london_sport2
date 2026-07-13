from pathlib import Path
import pandas as pd
import numpy as np
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import seaborn as sns


def proxy_suitability():
    data_path = ROOT / 'exploration'/ 'data'/ 'boroughs'/ 'cleaned'/ 'borough_profiles.csv'
    save_path = Path('C:/Users/fergu/Documents/GitHub/london_sport2/figures') / 'corr_plot.png'
    df = pd.read_csv(data_path)
    df = df.dropna()
    of_interest = ['VolAny', 'comm1', 'IMD10', 'status']
    corr = df[of_interest].corr()
    new_corr = corr.iloc[1:,:-1]
    mask = np.triu(np.ones_like(new_corr), k=1)
    print(corr)
    print(new_corr)
    sns.heatmap(
        new_corr, 
        annot=True,
        mask=mask,
        square=True,
        cmap='OrRd',
        linewidth=0.5
    )
    
    plt.savefig(save_path)
    print(f'Figure saved to: {save_path}')
    plt.show()
proxy_suitability()
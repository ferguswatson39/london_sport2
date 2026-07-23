import numpy as np
import pandas as pd
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import matplotlib.pyplot as plt
import sys
import seaborn as sns

SPORT_MAP = {
    'WALKALL' : 'Walking',     
    'CYCALL' : 'Cycling',
    'RUNATHMULTI' : 'Running',
    'ADVWATERSPORT' : 'Water Sports',
    'TEAMSPORT' : 'Team Sports',
    'DANCEALL' : 'Dance/Gymnastics',
    'FOOTBALL' : 'Football',
    'RACKETSPORT' : 'Racket Sports',
    'COMBATTARGET' : 'Combat Sports',
    'WINTER' : 'Winter Sports',
    'CRICKET' : 'Cricket',
    'BASKETBALL' : 'Basketball',
    'GYMTRAMPCHEER' : 'Gymnastics',
    'RUGBYUNION' : 'Rugby',
    'NETBALL' : 'Netball',
    'HOCKEY' : 'Hockey'
}

CATEGORY_MAP = {
    'Walking' : 'Active Travel',     
    'Cycling' : 'Active Travel',
    'Running' : 'Active Travel',
    'Water Sports' : 'Water Sports',
    'Dance/Gymnastics' : 'Dance/Gymnastics',
    'Racket Sports' : 'Racket Sports',
    'Combat Sports' : 'Combat Sports',
    'Winter Sports' : 'Winter Sports',
    'Team Sports' : 'Team Sports'
}

def prepare_sporting_distributions(threshold : float = 0.5):
    df_path = Path(ROOT / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    df = pd.read_csv(df_path)
    MEMS_COLS = [col for col in df.columns if col.startswith('MEMS7_')]
    total_mems = (df['MEMS7_ALL'] * df['wt_final']).sum()
    results = []
    for col in MEMS_COLS:
        mems_contribution = (df[col].fillna(0) * df['wt_final']).sum() 
        results.append({'Sport' : col, 'MEMS Contribution %' : mems_contribution / total_mems * 100})
    distribution_df = pd.DataFrame(results).sort_values(by = 'MEMS Contribution %', ascending = False).reset_index(drop = True)
    distribution_df = distribution_df[distribution_df['MEMS Contribution %'] >= threshold]
    exclude = ['MEMS7_ALL', 'A0', 'B0', 'WALKALLRUN', 'ACTTRAV', 'LEISURE', 'ROLLER', 'F']
    distribution_df = distribution_df[~ distribution_df['Sport'].str.contains('|'.join(exclude))]
    distribution_df['Sport'] = distribution_df['Sport'].str.split('_').str[1].replace(SPORT_MAP)
    distribution_df['Category'] = distribution_df['Sport'].replace(CATEGORY_MAP)
    distribution_df.loc[distribution_df['Sport'] == 'Dance/Gymnastics', 'MEMS Contribution %'] += distribution_df.loc[distribution_df['Sport'] == 'Gymnastics', 'MEMS Contribution %'].sum()
    distribution_df = distribution_df[distribution_df['Sport'] != 'Gymnastics']
    return distribution_df

distribution_df = prepare_sporting_distributions()
sns.barplot(distribution_df, x = 'MEMS Contribution %', y = 'Sport', hue = 'Category')
plt.legend('', frameon = False)
plt.tight_layout()
plt.ylabel(None)
plt.savefig('figures/specific-sports', bbox_inches = 'tight', dpi = 500)
plt.show()



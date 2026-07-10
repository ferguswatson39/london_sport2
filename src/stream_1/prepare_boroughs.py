from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_geographic_data
from src.loading_data.data_catalogue import DataCatalogue

def prepare_borough_data():
    borough_df = get_geographic_data()
    borough_df = borough_df.groupby(['LA_2023', 'LA_Name', 'year'])['MEMS7_ALL'].mean()
    borough_df = borough_df.reset_index()
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({'Kingston upon Thames': 'Kingston', 'Richmond upon Thames': 'Richmond'})
    borough_df = borough_df[borough_df['LA_2023'] != 59.0] # Removing City of London (COVID Outliers)
    return borough_df

def prepare_national_data():
    national_df = get_geographic_data()
    national_df = national_df.groupby('year')['MEMS7_ALL'].mean()
    national_df = national_df.reset_index()
    return national_df

def prepare_grouped_boroughs():
    dc = DataCatalogue()
    df_path = Path(ROOT / 'exploration' / 'data' / 'master_data' / '2016_to_2023_full_preprocessed_data_set.csv.gz')
    raw = pd.read_csv(df_path)
    raw['LA_Name'] = raw['LA_2023'].map(dc.get_data_dict()['LA_2023']['value_labels'])
    borough_df = raw.groupby(['LA_2023', 'LA_Name', 'year']).agg(
        MEMS7_ALL=('MEMS7_ALL', 'mean'),
        IMD_mean=('IMD10', 'mean'),
        LondInOut=('LondInOut', 'first')
    ).reset_index()
    borough_df['LA_Name'] = borough_df['LA_Name'].replace({
        'Kingston upon Thames': 'Kingston',
        'Richmond upon Thames': 'Richmond'
    })
    borough_df['LondInOut'] = borough_df['LondInOut'].map({1.0: 'Outer', 2.0: 'Inner'})
    borough_df = borough_df[borough_df['LA_2023'] != 59.0]
    print('DataFrame Cleaned Successfully...')
    print(f'>>> Columns:\n{borough_df.columns.tolist()}')
    print(f'>>> Shape {borough_df.shape}')
    return borough_df

# Initial Visualisation
if __name__ == "__main__":
    borough_df = prepare_borough_data()
    g = sns.relplot(kind = 'scatter', 
                data = borough_df, 
                x = 'year', 
                y = 'MEMS7_ALL', 
                col = 'LA_Name', 
                col_wrap = 8,
                height = 2, 
                aspect = 1.4,
                s = 50,
                palette = 'RdYlGn',
                hue = 'MEMS7_ALL',
                legend = False,
                edgecolor = 'grey', 
                linewidth = 1)
    g.set_titles('{col_name}', weight = 'bold')
    g.set(xlabel = 'Year', ylabel = 'Avg Sport Participation')
    g.set_xticklabels([])
    g.set_yticklabels([])
    plt.savefig('src/stream_1/figures/Average Participation Boroughs', bbox_inches = 'tight')
    plt.show()
    
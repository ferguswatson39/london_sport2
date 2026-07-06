from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.loading_data.load_data import get_borough_data

df = get_borough_data()
boroughs = df.groupby(['LA_2023', 'LA_Name', 'year'])['MEMS7_ALL'].mean()
boroughs = boroughs.reset_index()
boroughs['LA_Name'] = boroughs['LA_Name'].replace({'Kingston upon Thames': 'Kingston', 'Richmond upon Thames': 'Richmond'})
boroughs = boroughs[boroughs['LA_2023'] != 59.0] # Removing City of London (COVID Outliers)

g = sns.relplot(kind = 'scatter', 
            data = boroughs, 
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
plt.show()
from pathlib import Path
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
import plotly.express as px

df_path = ROOT / 'data' /'master_data'/ '2016_to_2023_clustering_output_data.csv'
save_path = ROOT / 'figures' / 'clustering' / 'cluster_comparison.png'

df = pd.read_csv(df_path)
df_freq = pd.crosstab(df['LCA_Class'], df['hdb_labels_281_50'], normalize='index')

df_freq.index, df_freq.columns = df_freq.index + 1, df_freq.columns + 1
threshold = 0.05 
masked_df_freq = df_freq[df_freq > threshold]

fig = px.imshow(masked_df_freq, 
                text_auto=".0%", 
                aspect="auto", 
                color_continuous_scale="Blues",
                labels = {'x' : 'HDBSCAN Cluster Labels', 'y': 'LCA Cluster Labels'})
fig.update_layout(coloraxis_showscale = False)
fig.update_xaxes(type='category', title_font = dict(weight='bold'))
fig.update_yaxes(type='category', title_font = dict(weight='bold'))
fig.write_image(save_path, scale = 5, width = 1080)
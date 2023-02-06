import glob
import pandas as pd

csv_files = glob.glob('train_road_labeled_*.{}'.format('csv'))
df_append = pd.DataFrame()
for file in csv_files:
            df_temp = pd.read_csv(file)
            df_append = df_append.append(df_temp, ignore_index=True)
df = df_append.drop(columns=df_append.columns[0])
df.to_csv('train_road_labeled.csv')
 
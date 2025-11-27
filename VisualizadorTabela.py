import pandas as pd

df = pd.read_csv("spotify_clean.csv")

ranges = df.describe().loc[['min', 'max']]
print(ranges)

import pandas as pd

from pybaseball import statcast
#stats = statcast(start_dt="2022-06-24", end_dt="2022-06-28")
stats = pd.read_pickle('stats.pickle')
print(stats.columns)
#stats.to_pickle('stats.pickle')
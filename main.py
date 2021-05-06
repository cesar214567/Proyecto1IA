import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
data.dropna(subset=['date','lat','lon','zone','deaths'])#falta infectados (hay varias filas )
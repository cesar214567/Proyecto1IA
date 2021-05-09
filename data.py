import numpy as np
import pandas as pd
import csv
from datetime import date

data = pd.read_csv('data.csv')

data= data.dropna(how='any',axis=0,subset=['date','lat','lon','zone','deaths','confirmed'])#falta infectados (hay varias filas )



def get_date_differential(given_date):
    initial_date=date(2020,3,6)
    #print(int(given_date[0:4]),int(given_date[5:7]),int(given_date[8:10]))
    new_date=date(int(given_date[0:4]),int(given_date[5:7]),int(given_date[8:10]))
    return (new_date-initial_date).days


with open('data2.csv','w+') as csv_file:
    zonas = {"ZONA CENTRO":0,"ZONA NORTE":1,"ZONA SUR":2}
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['date','lat','lon','zone','deaths','confirmed'])
    for index,row in data.iterrows():
        writer.writerow([get_date_differential(row['date']),row['lat'],row['lon'],zonas[row['zone']],row['deaths'],row['confirmed']])
        print(get_date_differential(row['date']))

print(data.zone.unique())
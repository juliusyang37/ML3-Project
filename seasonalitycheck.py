"""
@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import events as eve

### Generate first table percentage of seasonality
dax = web.get_data_yahoo('^GDAXI', start='2000-01-01', end='2010-12-31')
dax.columns = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj']
fridays = eve.get_fridays()
firsts = eve.get_first_days()
hol = eve.get_market_holidays()
daxR = dax.Adj.values[1:]/dax.Adj.values[0:-1]-1
alldays = (daxR>0).mean()
rfri = []
rhol = []
rfirsts = []


for d in fridays:
    try:
        rfri.append(dax.loc[d+dt.timedelta(3),'Close']/dax.loc[pd.to_datetime(d+dt.timedelta(-1)), 'Close']-1)
    except:
        pass

for d in firsts:
    try:
        rfirsts.append(dax.loc[np.busday_offset((d+dt.timedelta(+5)).date(), 0, roll='backward'),'Close']/dax.loc[np.busday_offset((d+dt.timedelta(-7)).date(), 0, roll='forward'),'Close']-1)
    except:
        pass 

for d in hol:
    try:
        rhol.append(dax.loc[np.busday_offset((d+dt.timedelta(2)).date(), 0, roll='forward'),'Close']/dax.loc[np.busday_offset((d+dt.timedelta(-1)).date(), 0, roll='backward'),'Close']-1)
            
    except:
        pass 

rfri = np.array(rfri)
rhol = np.array(rhol)
rfirsts=np.array(rfirsts) 
print((rfri>0).mean())
print((rhol>0).mean())
print((rfirsts>0).mean())
print((daxR>0).mean())

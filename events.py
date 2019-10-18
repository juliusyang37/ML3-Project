"""
@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import numpy as np
import pandas as pd
import pandas_datareader as web
from datetime import date
import datetime as dt
import FeatureGeneration as fg

#### Code to retrieve holiday outside of week ends ####
def get_market_holidays(start='2000-01-01', end='2019-01-01'):
    temp = web.get_data_yahoo('^GDAXI', start=start, end=end)
    temp = temp.loc[temp.Volume>0,:]
    opens=set(temp.index.values)
    opens = pd.to_datetime(list(opens))
    weekmask = [1, 1, 1, 1, 1, 0, 0]
    busdays = np.arange(start, end, dtype='datetime64[D]')
    busdays = pd.to_datetime(busdays[np.is_busday(busdays, weekmask=weekmask)])
    holidays = set(busdays).difference(opens)
    df = pd.to_datetime(list(holidays)).sort_values()
    return df


#### All of the week ends
def get_fridays(start='2000-01-01', end='2019-01-01'):
    weekmask = [0, 0, 0, 0, 1, 0, 0]
    fridays = np.arange(start, end, dtype='datetime64[D]')
    fridays = pd.to_datetime(fridays[np.is_busday(fridays, weekmask=weekmask)]).sort_values()
    return fridays

### Begining of each month
def get_first_days(start='2000-01-01', end='2019-01-01'):
    days = pd.date_range(start, end, freq='MS')
    return days

### Get random tickers
def get_random_tickers(n, ticklist):
    if n >len(ticklist):
        raise Exception('n is bigger than ticklist')
    return np.random.choice(ticklist, size=n, replace=False)

### Generate trading days
def get_model_days(start='2000-01-01', end='2020-01-01', tocsv=True):
    fridays = get_fridays(start, end)
    firsts = get_first_days(start, end)
    hol = get_market_holidays(start, end)
    holbuy = hol.map(lambda x : x + dt.timedelta(-1))
    holsell = hol.map(lambda x : x + dt.timedelta(+2))
    firstsbuy = firsts.map(lambda x : x + dt.timedelta(-1))
    firstssell = firsts.map(lambda x : x + dt.timedelta(+2))
    mondays = fridays.map(lambda x : x + dt.timedelta(+3))
    df1 = pd.DataFrame([holbuy.values, holsell.values, ['hol']*len(hol)]).transpose()
    df2 = pd.DataFrame([firstsbuy.values, firstssell.values, ['firsts']*len(firsts)]).transpose()
    df3 = pd.DataFrame([fridays.values, mondays.values, ['fridays']*len(fridays)]).transpose()
    temp = pd.concat([df1, df2, df3], axis=0, join='outer', join_axes=None, ignore_index=True,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
    temp.columns=['Buy', 'Sell', 'Type']
    temp.loc[temp.Type=='hol', 'Type'] = 1
    temp.loc[temp.Type=='firsts', 'Type'] = 2
    temp.loc[temp.Type=='fridays', 'Type'] = 3
    temp.reset_index(drop=True, inplace=True)
    temp.to_csv('trading_days.csv')
    '''
    dtes=pd.DataFrame(index=range(len(temp.Buy.unique())), columns=['Buy', 'Sell', 'Hols', 'Firsts', 'Fridays'])
    dtes.Buy=temp.Buy.unique()
    dtes.Sell = temp.Sell.unique()
    for i in range(len(dtes.index)):
        if dtes.Buy[i] in holbuy : dtes.loc[i,'Hols'] = 1
        else: dtes.loc[i,'Hols'] = 0
        if dtes.Buy[i] in fridays : dtes.loc[i,'Fridays'] = 1
        else: dtes.loc[i,'Fridays'] = 0
        if dtes.Buy[i] in firstsbuy : dtes.loc[i,'Firsts'] = 1
        else: dtes.loc[i,'Firsts'] = 0
    '''
    if tocsv: temp.to_csv('trading_days.csv')
    return temp

def get_model_days_2(start='2000-01-01', end='2020-01-01', tocsv=True):
    fridays = get_fridays(start, end)
    firsts = get_first_days(start, end)
    hol = get_market_holidays(start, end)
    holbuy = hol.map(lambda x : np.busday_offset((x+dt.timedelta(-1)).date(), 0, roll='backward'))
    holsell = hol.map(lambda x : np.busday_offset((x+dt.timedelta(+2)).date(), 0, roll='forward'))
    firstsbuy = firsts.map(lambda x : np.busday_offset((x+dt.timedelta(-7)).date(), 0, roll='forward'))
    firstssell = firsts.map(lambda x : np.busday_offset((x+dt.timedelta(5)).date(), 0, roll='backward'))
    mondays = fridays.map(lambda x : x + dt.timedelta(+3))
    df1 = pd.DataFrame([holbuy.values, holsell.values, ['hol']*len(hol)]).transpose()
    df2 = pd.DataFrame([firstsbuy.values, firstssell.values, ['firsts']*len(firsts)]).transpose()
    df3 = pd.DataFrame([fridays.values, mondays.values, ['fridays']*len(fridays)]).transpose()
    temp = pd.concat([df1, df2, df3], axis=0, join='outer', join_axes=None, ignore_index=True,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
    temp.columns=['Buy', 'Sell', 'Type']
    temp.loc[temp.Type=='hol', 'Type'] = 1
    temp.loc[temp.Type=='firsts', 'Type'] = 2
    temp.loc[temp.Type=='fridays', 'Type'] = 3
    temp.reset_index(drop=True, inplace=True)
    dtes=pd.DataFrame(index=range(len(temp.Buy.unique())), columns=['Buy', 'Sell', 'Hols', 'Firsts', 'Fridays'])
    dtes.Buy=temp.Buy.unique()
    dtes.Sell = temp.Sell.unique()
    temp.to_csv('trading_days.csv')

    '''
    return temp
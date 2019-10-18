"""
@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import numpy as np
import pandas as pd
import pandas_datareader as web


def SMA(data, n, i): 
   #data is a series (1 column of the data frame)
   if i >= n:                                                
       return data[i-n:i+1].mean()
   else:
       return data[0:i+1].mean()
   
def EMA(data, n, i,ema1):
   return data[i]*(2/(n+1))+ema1*(1-2/(n+1)) 


def update_tickers(ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
                              'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
                              '1COV.DE', 'DB1.DE', 'DPW.DE', 'IFX.DE', 'VNA.DE', 'WDI.DE'], start1='2000-01-01', start2='2019-01-01', end1='2000-01-05', end2='2019-01-01'):
    ### Load the tickers
    a = dict()
    b = dict()
    
    for tick in ticklist:
        try:
            temp = web.get_data_yahoo(tick, start=start1, end=end1)
            a[tick] = temp
        except:
            print(tick, 'did not work')
        try : 
            temp = web.get_data_yahoo(tick, start=start2, end=end2)
            b[tick] = temp
        except:
            print(tick, 'did not work')
            
    ticklist = set(a.keys()).intersection(set(b.keys()))
    
    ### These are the DAX tickers that exist today and also existed in 2000 and are the tickers we will be using ###
    '''
    new ticklist:
    
    ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
     'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE']
    '''
    return ticklist

def tick_data(tick, startdate, enddate, delta=0.94, tocsv=False, norm=False):
    a = web.get_data_yahoo(tick, start=startdate, end=enddate)
    a.iloc[:,5] = a.iloc[:,3]
    a = a.loc[a.Volume>0,:]
    b=a.copy()
    if norm:
        m = 100/a.iloc[0,2]
        a.iloc[:,0] = a.iloc[:,0]*m
        a.iloc[:,1] = a.iloc[:,1]*m
        a.iloc[:,2] = a.iloc[:,2]*m
        a.iloc[:,3] = a.iloc[:,3]*m
        a.iloc[:,5] = a.iloc[:,5]*m
    df = pd.DataFrame(columns = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'Return',
                               'Variance', 'ValueAtRisk', 'VarScalar', 'SMA20', 'SMA26', 'SMA32',
                               'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                               'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                               'ROC22', 'EMA12', 'EMA18', 'EMA24', 'EMA30', 'MACD1812', 'MACD2412',
                               'MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 'RSI8', 'RSI14',
                               'RSI20', 'PriceUp', 'PriceDown', 'SMA8Up', 'SMA8Down', 'SMA14Up',
                               'SMA14Down', 'SMA20Up', 'SMA20Down', 'Date', 'ADL', 'OBV', 'EMADL3',
                               'EMADL10', 'EMAHL10', 'EMAHL16', 'EMAHL22', 'CHV1010', 'CHV1016',
                               'CHV1022', 'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18',
                               'SlowD12', 'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24',
                               'SlowD24', 'CHO','High','Low','Open','Close','Volume','AdjClose', 'Ticker'], 
                      index = range(len(a.index)))

    df['Date'] = a.index

    ### EMA ###
    df.loc[0,'EMA10'] = a.iloc[0,5] #regular EMA 10
    df.loc[0,'EMA16'] = a.iloc[0,5] #regular EMA 16
    df.loc[0,'EMA22'] = a.iloc[0,5] #regular EMA 22
    df.loc[0,'EMA12'] = a.iloc[0,5] #regular EMA 12
    df.loc[0,'EMA18'] = a.iloc[0,5] #regular EMA 18
    df.loc[0,'EMA24'] = a.iloc[0,5] #regular EMA 24
    df.loc[0,'EMA30'] = a.iloc[0,5] #regular EMA 30

    ### MACD ###
    df.loc[0,'MACD1812'] = 0
    df.loc[0,'MACD2412'] = 0
    df.loc[0,'MACD3012'] = 0

    ### MACDS ###
    df.loc[0,'MACDS18129'] = 0
    df.loc[0,'MACDS24129'] = 0
    df.loc[0,'MACDS30129'] = 0

    ### SMA ###
    df.loc[0,'SMA10'] = a.iloc[0,5]
    df.loc[0,'SMA16'] = a.iloc[0,5]
    df.loc[0,'SMA22'] = a.iloc[0,5]
    df.loc[0,'SMA20'] = a.iloc[0,5]
    df.loc[0,'SMA26'] = a.iloc[0,5]
    df.loc[0,'SMA32'] = a.iloc[0,5]

    ### Returns ###
    df.loc[1:,'Return'] = (a.iloc[1:,5].values-a.iloc[0:-1,5].values)/a.iloc[0:-1,5].values


    ### Price Up/Down Vectors ###
    df.loc[1:, 'PriceUp'] = np.where((a.iloc[1:,5].values-a.iloc[0:-1,5].values)>=0,(a.iloc[1:,5].values-a.iloc[0:-1,5].values), np.nan)
    df.loc[1:, 'PriceDown'] = np.where((a.iloc[1:,5].values-a.iloc[0:-1,5].values)<0,-1*(a.iloc[1:,5].values-a.iloc[0:-1,5].values), np.nan)

    ### Variance ###

    df.loc[0,'Variance'] = np.power(df.iloc[1:32,6].std(),2) 

    ### Value at Risk ###

    df.loc[0,'ValueAtRisk'] = 1.96*np.sqrt(df.loc[0,'Variance'])
    df.loc[0,'VarScalar'] = 1

    ### ADL ###
    df.loc[0,'ADL'] = a.iloc[0,4]*\
                    ((a.iloc[0,3] - a.iloc[0,1])-(a.iloc[0,0]-a.iloc[0,3]))/\
                    (a.iloc[0,0]-a.iloc[0,1])
    if np.isnan(df.loc[0,'ADL']) : df.loc[0,'ADL']=0

    ### OBV ###
    df.loc[0,'OBV'] = a.iloc[0,4]

    ### EMADL ###
    df.loc[0,'EMADL3'] = df.loc[0,'ADL']
    df.loc[0,'EMADL10'] = df.loc[0,'ADL']

    ### EMAHL ### 
    df.loc[0,'EMAHL10'] = a.iloc[0,0]-a.iloc[0,1]
    df.loc[0, 'EMAHL16'] = a.iloc[0,0]-a.iloc[0,1]
    df.loc[0, 'EMAHL22'] = a.iloc[0,0]-a.iloc[0,1]

    for i in range(1, len(a.index)):
        
        ### EMA ###
        df.loc[i,'EMA10'] = EMA(a.iloc[:,5].values,10,i,df.loc[i-1,'EMA10'])
        df.loc[i,'EMA16'] = EMA(a.iloc[:,5].values,16,i,df.loc[i-1,'EMA16'])
        df.loc[i,'EMA22'] = EMA(a.iloc[:,5].values,22,i,df.loc[i-1,'EMA22'])
        df.loc[i,'EMA12'] = EMA(a.iloc[:,5].values,12,i,df.loc[i-1,'EMA12'])
        df.loc[i,'EMA18'] = EMA(a.iloc[:,5].values,18,i,df.loc[i-1,'EMA18'])
        df.loc[i,'EMA24'] = EMA(a.iloc[:,5].values,24,i,df.loc[i-1,'EMA24'])
        df.loc[i,'EMA30'] = EMA(a.iloc[:,5].values,30,i,df.loc[i-1,'EMA30'])

        ### MACD ###

        df.loc[i,'MACD1812'] = df.loc[i,'EMA18'] - df.loc[i,'EMA12']
        df.loc[i,'MACD2412'] = df.loc[i,'EMA24'] - df.loc[i,'EMA12']
        df.loc[i,'MACD3012'] = df.loc[i,'EMA30'] - df.loc[i,'EMA12']

        ### MACDS ###

        df.loc[i,'MACDS18129'] = EMA(df.loc[0:i+1,'MACD1812'],9,i,df.loc[i-1,'MACDS18129'])
        df.loc[i,'MACDS24129'] = EMA(df.loc[0:i+1,'MACD2412'],9,i,df.loc[i-1,'MACDS24129'])
        df.loc[i,'MACDS30129'] = EMA(df.loc[0:i+1,'MACD3012'],9,i,df.loc[i-1,'MACDS30129'])

        ### SMA ###
        df.loc[i,'SMA10'] = SMA(a.iloc[:,5].values,10,i) #SMA n=10
        df.loc[i,'SMA16'] = SMA(a.iloc[:,5].values,16,i) #SMA n=16
        df.loc[i,'SMA22'] = SMA(a.iloc[:,5].values,22,i) #SMA n=22
        df.loc[i,'SMA20'] = SMA(a.iloc[:,5].values,20,i) #SMA n=20
        df.loc[i,'SMA26'] = SMA(a.iloc[:,5].values,26,i) #SMA n=26
        df.loc[i,'SMA32'] = SMA(a.iloc[:,5].values,32,i) #SMA n=32

        ### SMA Price Up/Down Vectors ###
        df.loc[i,'SMA8Up'] = SMA(df.loc[:,'PriceUp'],8,i) #SMA n=8
        df.loc[i,'SMA8Down'] = SMA(df.loc[:,'PriceDown'],8,i) #SMA n=8
        df.loc[i,'SMA14Up'] = SMA(df.loc[:,'PriceUp'],14,i) #SMA n=14
        df.loc[i,'SMA14Down'] = SMA(df.loc[:,'PriceDown'],14,i) #SMA n=14
        df.loc[i,'SMA20Up'] = SMA(df.loc[:,'PriceUp'],20,i) #SMA n=20
        df.loc[i,'SMA20Down'] = SMA(df.loc[:,'PriceDown'],20,i) #SMA n=20


        ### Variance ###
        df.loc[i,'Variance'] = delta*df.loc[i-1,'Variance']+(1-delta)*np.power(df.loc[i,'Return'],2)

        ### Value at Risk ###
        if (df.loc[i,'Return'] < -df.loc[i-1,'ValueAtRisk']):
            df.loc[i,'VarScalar'] = df.loc[i-1,'VarScalar'] + 1
        else:
            df.loc[i,'VarScalar'] = df.loc[i-1,'VarScalar']*delta

        df.loc[i,'ValueAtRisk'] = 1.96*df.loc[i,'VarScalar']*np.sqrt(df.loc[i,'Variance']) 

        ### Bollu/Bolld ###
        if i>=20:
            df.loc[i,'Bollu20'] = df.loc[i,'SMA20']+2*a.iloc[i-20:i+1,5].std()
            df.loc[i,'Bolld20'] = df.loc[i,'SMA20']-2*a.iloc[i-20:i+1,5].std()
        else:
            df.loc[i,'Bollu20'] = df.loc[i,'SMA20']+2*a.iloc[0:i+1,5].std()
            df.loc[i,'Bolld20'] = df.loc[i,'SMA20']-2*a.iloc[0:i+1,5].std()
        if i>=26:
            df.loc[i,'Bollu26'] = df.loc[i,'SMA26']+2*a.iloc[i-26:i+1,5].std()
            df.loc[i,'Bolld26'] = df.loc[i,'SMA26']-2*a.iloc[i-26:i+1,5].std()
        else:
            df.loc[i,'Bollu26'] = df.loc[i,'SMA26']+2*a.iloc[0:i+1,5].std()
            df.loc[i,'Bolld26'] = df.loc[i,'SMA26']-2*a.iloc[0:i+1,5].std()
        if i>=32:
            df.loc[i,'Bollu32'] = df.loc[i,'SMA32']+2*a.iloc[i-32:i+1,5].std()
            df.loc[i,'Bolld32'] = df.loc[i,'SMA32']-2*a.iloc[i-32:i+1,5].std()
        else:
            df.loc[i,'Bollu32'] = df.loc[i,'SMA32']+2*a.iloc[0:i+1,5].std()
            df.loc[i,'Bolld32'] = df.loc[i,'SMA32']-2*a.iloc[0:i+1,5].std()


        ### ADL ###   
        df.loc[i, 'ADL'] = df.loc[i-1, 'ADL'] + a.iloc[i,4]*\
                        ((a.iloc[i,3] - a.iloc[i,1])-(a.iloc[i,0]-a.iloc[i,3]))/\
                        (a.iloc[i,0]-a.iloc[i,1])
        if np.isnan(df.loc[i, 'ADL']) : df.loc[i, 'ADL'] = df.loc[i-1, 'ADL']
        if a.iloc[i,0] == a.iloc[i,1] : df.loc[i, 'ADL'] = df.loc[i-1, 'ADL']
        
        ### OBV ###
        if a.iloc[i,5]-a.iloc[i-1,5]>0:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] + a.iloc[i,4]
        else:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] - a.iloc[i,4]

        ### EMA ADL ###
        df.loc[i, 'EMADL3'] = EMA(df.ADL.values,3,i,df.loc[i-1,'EMADL3'])
        df.loc[i, 'EMADL10'] = EMA(df.ADL.values,10,i,df.loc[i-1,'EMADL10'])

        ### EMAHL / CHV ### 
        df.loc[i,'EMAHL10'] = EMA(a.iloc[:,0]-a.iloc[:,1], 10, i, df.loc[i-1,'EMAHL10'])
        df.loc[i, 'EMAHL16'] = EMA(a.iloc[:,0]-a.iloc[:,1], 16, i, df.loc[i-1,'EMAHL16'])
        df.loc[i, 'EMAHL22'] = EMA(a.iloc[:,0]-a.iloc[:,1], 22, i, df.loc[i-1,'EMAHL22'])
        if i>=10:
            df.loc[i, 'CHV1010'] = df.loc[i,'EMAHL10']/df.loc[i-10,'EMAHL10']-1
            df.loc[i, 'CHV1016'] = df.loc[i,'EMAHL16']/df.loc[i-10,'EMAHL16']-1
            df.loc[i, 'CHV1022'] = df.loc[i,'EMAHL22']/df.loc[i-10,'EMAHL22']-1

        ### fast % k ###
        if i >= 12 : df.loc[i,'FastK12'] = 100*(a.iloc[i,3]-a.iloc[i-12:i+1,1].min())/\
                                            (a.iloc[i-12:i+1,0].max() - a.iloc[i-12:i+1,1].min())
        if i >= 18: df.loc[i,'FastK18'] = 100*(a.iloc[i,3]-a.iloc[i-18:i+1,1].min())/\
                                            (a.iloc[i-18:i+1,0].max() - a.iloc[i-18:i+1,1].min())
        if i >= 24 : df.loc[i,'FastK24'] = 100*(a.iloc[i,3]-a.iloc[i-24:i+1,1].min())/\
                                            (a.iloc[i-24:i+1,0].max() - a.iloc[i-24:i+1,1].min())


        ### fastD ###
        if i >= 15 : df.loc[i,'FastD12'] = SMA(df.loc[:,'FastK12'], 3, i)
        if i >= 21: df.loc[i,'FastD18'] = SMA(df.loc[:,'FastK12'], 3, i)
        if i >= 27 : df.loc[i,'FastD24'] = SMA(df.loc[:,'FastK12'], 3, i)

        ### slow K ###
        if i >= 18 : df.loc[i,'SlowK12'] = SMA(df.loc[:,'FastD12'], 3, i)
        if i >= 24: df.loc[i,'SlowK18'] = SMA(df.loc[:,'FastD12'], 3, i)
        if i >= 30 : df.loc[i,'SlowK24'] = SMA(df.loc[:,'FastD12'], 3, i)

        ### Slow D ### 
        if i >= 21 : df.loc[i,'SlowD12'] = SMA(df.loc[:,'SlowK12'], 3, i)
        if i >= 27: df.loc[i,'SlowD18'] = SMA(df.loc[:,'SlowK12'], 3, i)
        if i >= 33 : df.loc[i,'SlowD24'] = SMA(df.loc[:,'SlowK12'], 3, i)


    ### CHO ###
    df.loc[:,'CHO'] = df.loc[:, 'EMADL3'] - df.loc[:, 'EMADL10']

    ### Mom ###

    df.loc[12:,'Mom12'] = a.iloc[12:,5].values-a.iloc[0:-12,5].values
    df.loc[18:,'Mom18'] = a.iloc[18:,5].values-a.iloc[0:-18,5].values
    df.loc[24:,'Mom24'] = a.iloc[24:,5].values-a.iloc[0:-24,5].values

    ### ACC ###

    df.loc[13:,'ACC12'] = df.iloc[13:,19].values-df.iloc[12:-1,19].values # ACC 12
    df.loc[19:,'ACC18'] = df.iloc[19:,20].values-df.iloc[18:-1,20].values # ACC 18
    df.loc[25:,'ACC24'] = df.iloc[25:,21].values-df.iloc[24:-1,21].values # ACC 24

    ### ROC ###
    df.loc[10:,'ROC10'] = 100*(a.iloc[10:,5].values-a.iloc[0:-10,5].values)/(a.iloc[0:-10,5].values)
    df.loc[16:,'ROC16'] = 100*(a.iloc[16:,5].values-a.iloc[0:-16,5].values)/(a.iloc[0:-16,5].values)
    df.loc[22:,'ROC22'] = 100*(a.iloc[22:,5].values-a.iloc[0:-22,5].values)/(a.iloc[0:-22,5].values)

    ### RSI ###

    df.loc[:,'RSI8']=100-100/(1+df.loc[:,'SMA8Up'].values/df.loc[:,'SMA8Down'].values)
    df.loc[df.loc[:,'SMA8Down'].isnull(),'RSI8'] = 100
    df.loc[df.loc[:,'SMA8Up'].isnull(),'RSI8'] = 0

    df.loc[:,'RSI14']=100-100/(1+df.loc[:,'SMA14Up'].values/df.loc[:,'SMA14Down'].values)
    df.loc[df.loc[:,'SMA14Down'].isnull(),'RSI14'] = 100
    df.loc[df.loc[:,'SMA14Up'].isnull(),'RSI14'] = 0

    
    df.loc[:,'RSI20']=100-100/(1+df.loc[:,'SMA20Up'].values/df.loc[:,'SMA20Down'].values)
    df.loc[df.loc[:,'SMA20Down'].isnull(),'RSI20'] = 100
    df.loc[df.loc[:,'SMA20Up'].isnull(),'RSI20'] = 0


    df.loc[:,'High'] = b.iloc[:,0].values
    df.loc[:,'Low'] = b.iloc[:,1].values
    df.loc[:,'Open'] = b.iloc[:,2].values
    df.loc[:,'Close'] = b.iloc[:,3].values
    df.loc[:,'Volume'] = b.iloc[:,4].values
    df.loc[:,'AdjClose'] = b.iloc[:,5].values
    df.loc[:, 'Ticker'] = tick
    if norm:
        df.loc[:,'Norm_High'] = a.iloc[:,0].values
        df.loc[:,'Norm_Low'] = a.iloc[:,1].values
        df.loc[:,'Norm_Open'] = a.iloc[:,2].values
        df.loc[:,'Norm_Close'] = a.iloc[:,3].values
        df.loc[:,'Norm_AdjClose'] = a.iloc[:,5].values
    
    if tocsv : 
        if norm: df.to_csv('tickDataNorm/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv')
        else: df.to_csv('tickData/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv')
    else: 
        return df



if __name__ == '__main__':
    ##Code to generate data
    ##Folders to create the data must exist - a main folder tickData and then folders for each delta
    ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
     'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE']    
    
    for dt in [0.9, 0.91, 0.92, 0.93, 0.94, 0.95,0.96]:
        for tick in ticklist:
            print('start', tick)
            tick_data(tick, '2000-01-01', '2020-01-01', delta=dt, tocsv=True, norm=True)
            print(tick, 'done')
            print()


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:37:49 2019

@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import neural_network as npy
import os 
import sys


print( os.path.dirname(os.path.realpath(__file__)) )
def get_random_tickers(n, ticklist):
    if n >len(ticklist):
        raise Exception('n is bigger than ticklist')
    return np.random.choice(ticklist, size=n, replace=False)
RSVE = []


# Our initial ticklist
ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE']

# The 7 tickers chosen randomly from this ticklist

ticklist = get_random_tickers(10, ticklist)
#ticklist = ['LIN.DE', 'HEN3.DE', 'DAI.DE', 'FME.DE', 'MUV2.DE', 'SIE.DE', 'DBK.DE']
ticklist = ['CON.DE', 'RWE.DE', 'DTE.DE', 'BEI.DE', 'HEI.DE', 'LIN.DE','FRE.DE', 'ADS.DE','FME.DE','MRK.DE']

tickdict = dict(zip(ticklist, range(1,len(ticklist)+1)))

## Paramteres
d0 = '2001-01-01' # begining of the waiting period
d1 = '2004-01-01' # end of the CV period - beg
d2 = '2006-01-01' # begining of the test period
d3 = '2008-01-01' # end of the test period

dcv1 = '2004-01-01'
dcv2 = '2005-01-01'


### Parameters
norm=True
cv_nlayers=True
cv_nneurones = False
lbd = 0.6 # history weight metric
alpha = 0.11 # risk management metric
cst = 10 #weight metric
delta = 0.91



nneurones = 9
nlayers = 2
drp = False
drpProb = 0.7
batch = False 
opt = 'RMSprop'
learning_rate = 0.002
n_epochs = 1000



# Generate the X matrix and Y matrix and make them have only trading days

dtes = pd.read_csv('trading_days.csv', index_col=0)
tempdt = dtes.copy()
tempdt.set_index('Buy', drop=True, inplace=True)
tempdt.index = pd.to_datetime(tempdt.index)

Y = []
X = pd.DataFrame(columns=['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 
                       'Ticker','Month', 'DAX', 'ADL', 'Type1', 'Type2', 'Type3', 'Y'])

dax = web.get_data_yahoo('^GDAXI', start=d0, end=d3)
prices = pd.DataFrame(columns=['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Date'])


for tick in ticklist:
    if norm:
        temp = pd.read_csv('tickDataNorm/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv', index_col=0)
    else:
        temp = pd.read_csv('tickData/'+ str(delta).replace('.', '') +'/' + tick.replace('.', '') +'.csv', index_col=0)
    prices = pd.concat([prices, temp.loc[:, ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Date']]],axis=0, ignore_index=True, sort=False,copy=True)
    # Select dates around events
    temp.set_index('Date', inplace=True, drop=True)
    temp.index = pd.to_datetime(temp.index)
    B = temp.loc[pd.to_datetime(dtes.Buy), 'Close']
    S = temp.loc[pd.to_datetime(dtes.Sell), 'Close']
    temp = temp.loc[tempdt.index, :]
    temp['Y'] = 100*(S.values-B.values)/B.values
    
    mask = np.logical_not(np.isnan(temp['Y'].values))
    temp = temp.loc[mask, :]
    if norm:
        temp.loc[:,'High'] = temp.loc[:,'Norm_High']
        temp.loc[:,'Low'] = temp.loc[:,'Norm_Low']
        temp.loc[:,'Open'] = temp.loc[:,'Norm_Open']
        temp.loc[:,'Close'] = temp.loc[:,'Norm_Close']
        temp.loc[:,'AdjClose'] = temp.loc[:,'Norm_AdjClose']
    #temp = temp.loc[pd.to_datetime(dtes.Buy), :]
    temp['Month'] = temp.index.month
    temp['Date'] = temp.index
    temp['DAX'] = dax.loc[temp.index, 'Adj Close']
    temp.loc[:,'Type'] = tempdt.loc[mask, 'Type'].values
    temp = temp.loc[:,['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL','Type', 'Y']]
    

    temp['Type1'] = (temp.loc[:,'Type'].values == 1)*1
    temp['Type2'] = (temp.loc[:,'Type'].values == 2)*1
    temp['Type3'] = (temp.loc[:,'Type'].values == 3)*1
    temp = temp.loc[:,['Date', 'EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 
                       'Ticker','Month', 'DAX', 'ADL', 'Type1', 'Type2', 'Type3', 'Y']]
    print(temp.Ticker.unique())
    X = pd.concat([X, temp], axis=0, ignore_index=True, copy=True, sort=False)


T = X.Ticker.unique()
tickdict = dict(zip(T, range(len(T))))
for tick in T:
    X.loc[X.Ticker==tick,'Ticker']= tickdict[tick]
    
X.sort_values(by=['Date', 'Ticker'], inplace=True)
X.set_index('Date', drop=True, inplace=True)
X = X.loc[((X.index>=pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d3))), :]

## X is generated

## Normalizing features
C = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                       'Month', 'DAX', 'ADL','Type1', 'Type2', 'Type3']

Cnorm = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22','ValueAtRisk', 
                       'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                       'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                       'ROC22', 'MACD1812', 'MACD2412','MACD3012', 'MACDS18129', 'MACDS24129', 'MACDS30129', 
                       'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016', 'CHV1022',
                       'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18', 'SlowD12',
                       'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24', 'SlowD24',
                       'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose',
                       'Month', 'DAX', 'ADL']


# Remove outliers
q1, q2 = X['Y'].quantile(0.98), X['Y'].quantile(0.02)
X = X[(X['Y'] < q1) & (X['Y'] > q2)]

## Normalization
Xtrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), C].copy()
Ytrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), 'Y'].copy()

Xmean = np.mean(Xtrain1.loc[:,Cnorm])
Xstdev = np.std(Xtrain1.loc[:,Cnorm])



X_old = X.copy()
X.loc[:, Cnorm] = (X.loc[:, Cnorm]-Xmean)/(Xstdev)

Xcv1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), C].copy()
Ycv1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), 'Y'].copy()
Xtrain1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), C].copy()
Ytrain1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), 'Y'].copy()

# Normalizing is done

#drp = True
#drpProb = 0.2
#learning_rate = 0.1
#opt='SGD'
#nneurones = 10

nneurones = 20
if cv_nlayers:
    #Code for cross validating number of layers - uses 50 neurones per layer
    train_loss = []
    nl = []
    for nlayers in range(8, 0, -1):
        print(nlayers)
        ls = [nneurones]*nlayers
        ls = np.insert(ls, 0, Xcv1.shape[1]).tolist()
        ls.append(1)
        #print(ls, drp, drpProb, batch, opt, learning_rate, n_epochs)
        nnt, err = npy.nnTrain(Xtrain1, Ytrain1, ls, drp, drpProb, batch, opt, learning_rate, n_epochs) 
        er  = npy.nnTest(nnt, Xcv1, Ycv1)
        train_loss.append(er)
        print(err[-1],er)
        #print('length',len(list(nnt.parameters())))
        #train_loss.append(err[-1])
        plt.plot(err)
        plt.show()
        
        nl.append(nlayers)
    plt.figure(figsize=(10,10))
    plt.scatter(nl, train_loss)
    plt.xlabel('Number of Layers')
    plt.ylabel('Training error')
    plt.title('Number of hidden layers cross validation')
    plt.show()
    sys.exit()
#--> 10 layers seem enough
nlayers = 3
cv_nneurones = True

if cv_nneurones:
    train_loss = []
    nl = []
    for nneurones in range(30, 0, -2):#[20,15, 10,5, 1]: #range(20, 0, -5):
        print(nneurones)
        ls = [nneurones]*nlayers
        ls = np.insert(ls, 0, Xcv1.shape[1]).tolist()
        ls.append(1)
        nnt, err = npy.nnTrain(Xtrain1, Ytrain1, ls, drp, drpProb, batch, opt, learning_rate, n_epochs)  
        er  = npy.nnTest(nnt, Xcv1, Ycv1)       
        train_loss.append(er) 
        
        plt.plot(err)
        nl.append(nneurones)
    plt.figure(figsize=(10,10))
    plt.scatter(nl, train_loss)
    plt.xlabel('Number of neurones per layer')
    plt.ylabel('Test error')
    plt.title('Number of neurones cross validation') 
    sys.exit()
    #---> 10 neurones seem enough


########################################################

### Add cross validation for nn paramters ##############

########################################################
    
 
    
    
## The Default value for vanilla NN
nlayers = 4
nneurones = 12
drp = False
drpProb = 0.1
batch = False
optimizer = 'SGD'
learning_rate = 0.002
n_epochs = 1000  

### The optimal parameters from grid search


nneurones = 9
nlayers = 2
drp = True
drpProb = 0.7
batch = False 
opt = 'RMSprop'
learning_rate = 0.002
n_epochs = 1000

#creation of list_index for the nn function
ls = [nneurones]*nlayers
ls = np.insert(ls, 0, Xcv1.shape[1]).tolist()
ls.append(1)
colused = C


## Train
#X.sort_values(by=['Date','Ticker'], inplace=True)

rfL = []
dtlist = X.index.unique().sort_values()
first_day= dtlist[0]
d = first_day+dt.timedelta(50)

#rfonerandomforest = RandomForestRegressor(n_estimators=200, max_features=None).fit(X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),colused],X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),'Y'])
rfonerandomforest = npy.nnTrain(X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),colused],X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),'Y'],
                                      ls, drp, drpProb, batch, opt, learning_rate, n_epochs)[0] #RandomForestRegressor(n_estimators=200, max_features=None).fit(X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),colused],X.loc[((X.index>=pd.to_datetime(d0)) & (X.index < pd.to_datetime(d1))),'Y'])
print('One train')
portfolioonerandomforest =  pd.DataFrame(columns=['Ponerandomforest'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())
yhatonerandomforest = pd.DataFrame(npy.nnPredict(rfonerandomforest, X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d3))),colused]),\
                                   index= X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d3))),:].index)
print('One predicted')
yhatonerandomforest['Ticker'] = X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d3))),'Ticker']
while d < pd.to_datetime(d3):
    X50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),colused]
    Y50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),'Y']
    rf = npy.nnTrain(X50_train, Y50_train, ls, drp, drpProb, batch, opt, learning_rate, n_epochs)[0]    
    rfL.append((d, rf))
    if d >= first_day + dt.timedelta(100):
        X100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),colused]
        Y100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),'Y']
        rf = npy.nnTrain(X100_train, Y100_train, ls, drp, drpProb, batch, opt, learning_rate, n_epochs)[0]
        rfL.append((d,rf))
    if d >= first_day + dt.timedelta(200):
        X200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),colused]
        Y200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),'Y']
        rf = npy.nnTrain(X200_train, Y200_train, ls, drp, drpProb, batch, opt, learning_rate, n_epochs)[0]
        rfL.append((d,rf))
    d = d+ dt.timedelta(50)




#Weights2

def calc_ki (lbd, ki1, ri):
    #return lbd*ri*100 + (1-lbd)*ki1
    return lbd/(ri) + (1-lbd)*ki1

def calc_wi(k, T):
    return  cst * k#/np.power(T, 0.5)
    #return max(0, 10-1/k)
w = np.zeros(len(rfL))
k = np.zeros(len(rfL))
r = np.zeros(len(rfL))
c = np.zeros(len(rfL))
mn = np.zeros(len(rfL))
rmse = np.ones(len(rfL))
mn[0]=1
w0= []
w1= []
w2= []
w3= []
w4= []
w5 = []
w6= []
w7= []
w8= []
w9 = []

nrf=0
feats = colused
d = dtlist[dtlist>=rfL[0][0]][0] 
prevT=1000
nrfprev = 0
P = []
Ponerandomforest = []
ntick = len(X.loc[:, 'Ticker'].unique())
B = np.zeros(ntick)
DL = pd.DataFrame(columns = range(ntick))
predictions = pd.DataFrame(columns = range(ntick))
predictionsnw = pd.DataFrame(columns = range(ntick))
portfolio = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())
portfolionw = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())
portfolionr = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())
portfoliobasic = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())

pt = pd.DataFrame(index =yhatonerandomforest.index.unique(), columns = yhatonerandomforest.Ticker.unique())


j=0

## The massive loop where we calculate the weights of the random forests and apply the risk management layer
while d < pd.to_datetime(d3):
    print(d)
    nrfprev = nrf
    T = d - first_day
    T50 = (T // 50).days
    if T50==1:
        nrf=1
    elif T50 == 2:
        nrf=3
    elif T50 == 3:
        nrf = 5
        prevT=3
    elif T50 > prevT:
        nrf +=3
        prevT = T50
    #if nrf==1:
    T1 = d - rfL[0][0]
    deltaT = len(dtlist[((dtlist<=d)&(dtlist>rfL[0][0]))])
    try:
        l = len(X.loc[d,feats].columns)
        k[0] = calc_ki(lbd, k[0], rmse[0])
        #rp = rfL[0][1].predict(X.loc[d,feats]) ##
        rp = npy.nnPredict(rfL[0][1], X.loc[d,feats])
        rmse[0] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y'])) ##
        if w[0] != 0 : w[0] = calc_wi(k[0], deltaT)
        else: 
            
            w[0] = 1#w[:nrfprev].mean()
            k[0] = 1/rmse[0]
        #rp = rfL[0][1].predict(X.loc[d,feats])
        BS =(rp / np.abs(rp).sum())
        #r[0] = ((BS * X.loc[d,'Y'].values/100).sum() - (1/len(X.loc[d,'Y'].values)) * (X.loc[d,'Y'].values/100).sum())
        
    except:
        d=pd.to_datetime(d)
        k[0] = calc_ki(lbd, k[0], rmse[0])
        #rp = rfL[0][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
        rp = npy.nnPredict(rfL[0][1], pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
        rmse[0] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']])) ##
        if w[0] != 0 : w[0] = calc_wi(k[0], deltaT)
        else: 
            
            w[0] = 1#w[:nrfprev].mean()
            k[0] = 1/rmse[0]
        #rp = rfL[0][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
        #r[0] = ((np.sign(rp) * X.loc[d,'Y']/100) - X.loc[d,'Y']/100)
        
    try:
        temp = np.zeros(ntick)
        #temp2 = rfL[0][1].predict(X.loc[d,feats].values)
        temp2 = npy.nnPredict(rfL[0][1], X.loc[d,feats])
        for tick in X.loc[d,'Ticker'].sort_values():
            temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
    except:
        #temp2 = rfL[0][1].predict(X.loc[d,feats].values.reshape(1, -1))
        temp2 = npy.nnPredict(rfL[0][1], X.loc[d,feats])
        temp[X.loc[d,'Ticker']] = temp2
    P.append(temp)
    
    for i in range(1, nrf):
        T1 = d - rfL[i][0]
        deltaT = len(dtlist[((dtlist<=d)&(dtlist>rfL[i][0]))])
        if i<nrfprev:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                
                k[i] = calc_ki(lbd, k[i], rmse[i])
                
                w[i] = calc_wi(k[i], deltaT)
                
                rp = npy.nnPredict(rfL[i][1], X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                rmse[i] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
            except:
                d=pd.to_datetime(d)
                
                k[i] = calc_ki(lbd, k[i], rmse[i])
                
                w[i] = calc_wi(k[i], deltaT)
                
                rp = npy.nnPredict(rfL[i][1], X.loc[d,feats])
                rmse[i] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
        else:
            try:
                l = len(X.loc[d,feats].columns)
                d=pd.to_datetime(d)
                mn[i] = w[:nrfprev].mean()
                w[i] = w[:nrfprev].mean()
                rp = npy.nnPredict(rfL[i][1],X.loc[d,feats])
                BS =(rp / np.abs(rp).sum())
                rmse[i] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
                k[i] = 1/rmse[i]
            except:
                d=pd.to_datetime(d)
                mn[i] = w[:nrfprev].mean()
                w[i] = w[:nrfprev].mean()
                rp = npy.nnPredict(rfL[i][1], X.loc[d,feats])
                #rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                rmse[i] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
                k[i] = 1/rmse[i]
        RSVE.append(rmse[i])
        try:
            temp = np.zeros(ntick)
            #temp2 = rfL[i][1].predict(X.loc[d,feats])
            temp2 = npy.nnPredict(rfL[i][1], pd.DataFrame(X.loc[d,feats]))
            for tick in X.loc[d,'Ticker'].sort_values():
                temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
        except:
            #temp2 = rfL[i][1].predict(X.loc[d,feats].values.reshape(1, -1))
            temp2 = npy.nnPredict(rfL[i][1], X.loc[d,feats])
            temp = np.zeros(ntick)
            temp[X.loc[d,'Ticker']] = temp2
        P.append(temp)
        
    if d >= pd.to_datetime(d1): 
        
        ## Applying our risk management mask
        
        P = pd.DataFrame(P)
        predictions.loc[d, :] = (P.values * np.array([w[:nrf].tolist()]*ntick).transpose()).sum(axis=0)/w.sum()
        predictionsnw.loc[d, :] = P.values .mean(axis=0)
        for tick in  range(ntick):
            temp = P.loc[:, tick]
            if (temp.values>0).sum()>0 : B[tick] = w[:nrf][temp.values>0].sum()/w.sum()
            else : B[tick]=0
        S = 1-B
        D = B-S
        D = np.abs(D)>alpha
        DL.loc[d, :] = D
        if j == 0 :
            c = 1
            cnr =1
            conerf = 1
            cnw = 1
            cbasic =1
        else : 
            c = portfolio.iloc[j-1,0]
            cnr = portfolionr.iloc[j-1,0]
            conerf = portfolioonerandomforest.iloc[j-1, 0]
            cnw = portfolionw.iloc[j-1, 0]
            cbasic = portfoliobasic.iloc[j-1, 0]
            
        try:
            len(yhatonerandomforest.loc[d,'Ticker'])
            pt.loc[d, yhatonerandomforest.loc[d,'Ticker'].unique()] = yhatonerandomforest.loc[d,:].groupby(['Ticker']).sum().values.flatten()
            pt.loc[d, pd.isnull(pt.loc[d,:])]=0
            wdonerf = pt.loc[d, :].values/pt.loc[d, :].abs().sum()
        except:
            pt.loc[d,:]=0
            wdonerf = np.zeros(ntick)
            
        wd = predictions.loc[d, :].values/predictions.loc[d, :].abs().sum()    
        wdnw = (P.values.mean(axis=0)) / np.abs(P.values.mean(axis=0)).sum()
        wdnr = wd
        ret = np.zeros(ntick)
        try:
            X.loc[d,:].Ticker.unique()
            for tick in X.loc[d,:].Ticker.unique():
                ret[tick] = X.loc[((X.index==d) & (X.Ticker==tick)),'Y'].values[0]/100
        except: 
            ret[X.loc[d,:].Ticker] = X.loc[d,:].Y/100
            
        ## Calculate the profit we earn from each variation of the model
        
        profit = (wd * c * ret * D).sum()
        profitnr = (wdnr * cnr * ret).sum()
        profitnw = (wdnw * cnr * ret*D).sum()
        profitonerf = (wdonerf *conerf * ret).sum()
        profitbasic = cbasic*ret.mean()
        portfolio.iloc[j, 0] = c+profit
        portfolionr.iloc[j,0] = cnr + profitnr # This is the profit for our trading strategy without equally weighted random forests and no risk management
        portfolionw.iloc[j,0] = cnw + profitnw
        portfolioonerandomforest.iloc[j,0] = conerf + profitonerf
        portfoliobasic.iloc[j,0] = cbasic + profitbasic
        j+=1
    
    w0.append(w[0])
    w1.append(w[1])
    w2.append(w[3])
    w3.append(w[5])
    w4.append(w[8])
    w5.append(w[11])
    w6.append(w[20])
    w7.append(w[29])
    w8.append(w[38])
    w9.append(w[47])

    P=[]
    try:
        d = dtlist[dtlist>d][0]
    except:
        break
              
## Plotting the evolution of our random forest weights over time
            
xa=range(len(w0))
plt.figure(figsize=(10,10))
plt.plot(xa, w1, label='2')

plt.xlabel('Time')
plt.ylabel('weight')
plt.title('Evolution of weights over time')
#plt.legend()
plt.show()

## Setting up the table of results and calculating each of the metrics

prices.Date = pd.to_datetime(prices.Date)    
retholdcv = np.zeros(ntick)
retholdtest = np.zeros(ntick)
prices.drop_duplicates(subset=['Date', 'Ticker'])
prices2 = prices.set_index('Date', drop=True)
stocks = pd.DataFrame(index = prices.Date.sort_values().unique(), columns=tickdict.keys())
XP = pd.DataFrame(index = pd.to_datetime(predictions.index), columns=predictions.columns)
X2 = X.reset_index()
X2 = X2.drop_duplicates(['Date', 'Ticker'])
X2 = X2.set_index('Date')
for tick in tickdict.keys():
    prices2.loc[prices2.Ticker==tick,'Ticker'] = tickdict[tick]
    prices.loc[prices.Ticker==tick,'Ticker'] = tickdict[tick]
    t2 = tickdict[tick]
    p1 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Close'].values[0]
    p2 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Open'].values[-1]
    p3 = prices.loc[((prices.Date>=pd.to_datetime(d2)) & (prices.Date<pd.to_datetime(d3)) & (prices.Ticker==t2)), 'Close'].values[0]
    p4 = prices.loc[((prices.Date>=pd.to_datetime(d2)) & (prices.Date<pd.to_datetime(d3)) & (prices.Ticker==t2)), 'Open'].values[-1]
    retholdcv[t2] = p2/p1-1
    retholdtest[t2] = p4/p3-1
    stocks.loc[:,tick] = prices2.loc[prices2.Ticker==t2,:].loc[stocks.index,'Close']
    temp = X2.loc[XP.index, :]
    XP.loc[temp.loc[temp.Ticker==t2,'Y'].index,t2] = temp.loc[temp.Ticker==t2,'Y'].values
    
xpval = XP.values
xpval[pd.isnull(xpval)]=0
XP = pd.DataFrame(xpval, index = XP.index, columns=XP.columns)
stocksreturn = pd.DataFrame(stocks.iloc[1:,:].values / stocks.iloc[0:-1,:].values, index=stocks.index[1:])-1
rs = (1 + stocksreturn.mean(axis=1)).values
portfolioBH = pd.DataFrame(np.cumprod(rs), index = pd.to_datetime(stocksreturn.index), columns=['P'])


holdcv = retholdcv.mean() 
holdtest = retholdtest.mean()   

dfcv = pd.DataFrame(index = ['Buy and Hold', 'One RF', 'Model', 'Model no weight', 'Model no risk management', 'Basic seasonality', 'Model with time decay'],
                    columns = ['Annualized return', 'Vol', 'Sharpe Ratio', 'RMSE'])

dftest = pd.DataFrame(index = ['Buy and Hold', 'One RF', 'Model', 'Model no weight', 'Model no risk management', 'Basic seasonality', 'Model with time decay'],
                    columns = ['Annualized return', 'Vol', 'Sharpe Ratio', 'RMSE'])

t0 = pd.to_datetime(d0)
t1 = pd.to_datetime(d1)
t2 = pd.to_datetime(d2)
t3 = pd.to_datetime(d3)
trd0 = dtlist[dtlist>=t0][0]
trd1 = dtlist[dtlist>=t1][0]
trd2 = dtlist[dtlist>=t2][0]
trd3 = dtlist[dtlist>=t2][-1]
meants = np.mean((dtlist[1:]-dtlist[0:-1]).days)
daxrcv = (dax.loc[dax.index[dax.index<t2][-1], 'Close']/dax.loc[dax.index[dax.index>=t1][0], 'Close'])**(1/(t2.year-t1.year))-1
daxrtest = (dax.loc[dax.index[dax.index<t3][-1], 'Close']/dax.loc[dax.index[dax.index>=t2][0], 'Close'])**(1/(t3.year-t2.year))-1

## Print the table of results for cross validation data
dfcv.loc['Buy and Hold', 'Annualized return'] = (1+holdcv)**(1/(t2.year-t1.year))-1
dfcv.loc['One RF', 'Annualized return'] = ((1+(portfolioonerandomforest.loc[trd2,:].values - portfolioonerandomforest.loc[trd1,:].values)/portfolioonerandomforest.loc[trd1,:].values)**(1/(t2.year-t1.year)))[0]-1
dfcv.loc['Model', 'Annualized return'] = ((1+(portfolio.loc[trd2,:].values - portfolio.loc[trd1,:].values)/portfolio.loc[trd1,:].values)**(1/(t2.year-t1.year)))[0]-1
dfcv.loc['Model no weight', 'Annualized return'] = ((1+(portfolionw.loc[trd2,:].values - portfolionw.loc[trd1,:].values)/portfolionw.loc[trd1,:].values)**(1/(t2.year-t1.year)))[0]-1
dfcv.loc['Model no risk management', 'Annualized return'] = ((1+(portfolionr.loc[trd2,:].values - portfolionr.loc[trd1,:].values)/portfolionr.loc[trd1,:].values)**(1/(t2.year-t1.year)))[0]-1
dfcv.loc['Basic seasonality', 'Annualized return'] = ((1+(portfoliobasic.loc[trd2,:].values - portfoliobasic.loc[trd1,:].values)/portfoliobasic.loc[trd1,:].values)**(1/(t2.year-t1.year)))[0]-1
dfcv.loc['One RF', 'Vol'] =  np.sqrt(365/meants)*((portfolioonerandomforest.loc[t1:t2,:].values[1:]/portfolioonerandomforest.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc['Model', 'Vol'] = np.sqrt(365/meants)*((portfolio.loc[t1:t2,:].values[1:]/portfolio.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc['Model no weight', 'Vol'] = np.sqrt(365/meants)*((portfolionw.loc[t1:t2,:].values[1:]/portfolionw.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc['Model no risk management', 'Vol'] = np.sqrt(365/meants)*((portfolionr.loc[t1:t2,:].values[1:]/portfolionr.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc['Basic seasonality', 'Vol'] = np.sqrt(365/meants)*((portfoliobasic.loc[t1:t2,:].values[1:]/portfoliobasic.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc['Buy and Hold', 'Vol'] = np.sqrt(365)*((portfolioBH.loc[t1:t2,:].values[1:]/portfolioBH.loc[t1:t2,:].values[0:-1])-1).std()
dfcv.loc[:, 'Sharpe Ratio'] = (dfcv.loc[:,'Annualized return'])/dfcv.loc[:,'Vol']
dfcv.loc['Model', 'RMSE'] = np.sqrt(mean_squared_error(predictions.loc[((predictions.index>=t1) &(predictions.index<t2)), :], XP.loc[((XP.index>=t1)& (XP.index<t2)),:]))
dfcv.loc['Model no risk management', 'RMSE'] = np.sqrt(mean_squared_error(predictions.loc[((predictions.index>=t1) &(predictions.index<t2)), :], XP.loc[((XP.index>=t1)& (XP.index<t2)),:]))
dfcv.loc['Model no weight', 'RMSE'] = np.sqrt(mean_squared_error(predictionsnw.loc[((predictionsnw.index>=t1) &(predictionsnw.index<t2)), :], XP.loc[((XP.index>=t1)& (XP.index<t2)),:]))
#dfcv.loc['One RF', 'RMSE'] = np.sqrt(mean_squared_error(rfonerandomforest.predict(X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d2))),colused]), 
#                                                        X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d2))),'Y']))
dfcv.loc['One RF', 'RMSE'] = np.sqrt(mean_squared_error(npy.nnPredict(rfonerandomforest, X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d2))),colused]), 
                                                        X.loc[((X.index>=pd.to_datetime(d1)) & (X.index < pd.to_datetime(d2))),'Y']))

## Print the table of results for test data
dftest.loc['Buy and Hold', 'Annualized return'] = (1+holdtest)**(1/(t3.year-t2.year))-1
dftest.loc['One RF', 'Annualized return'] = ((1+(portfolioonerandomforest.loc[trd3,:].values - portfolioonerandomforest.loc[trd2,:].values)/portfolioonerandomforest.loc[trd2,:].values)**(1/(t3.year-t2.year)))[0]-1
dftest.loc['Model', 'Annualized return'] = ((1+(portfolio.loc[trd3,:].values - portfolio.loc[trd2,:].values)/portfolio.loc[trd2,:].values)**(1/(t3.year-t2.year)))[0]-1
dftest.loc['Model no weight', 'Annualized return'] = ((1+(portfolionw.loc[trd3,:].values - portfolionw.loc[trd2,:].values)/portfolionw.loc[trd2,:].values)**(1/(t3.year-t2.year)))[0]-1
dftest.loc['Model no risk management', 'Annualized return'] = ((1+(portfolionr.loc[trd3,:].values - portfolionr.loc[trd2,:].values)/portfolionr.loc[trd2,:].values)**(1/(t3.year-t2.year)))[0]-1
dftest.loc['Basic seasonality', 'Annualized return'] = ((1+(portfoliobasic.loc[trd3,:].values - portfoliobasic.loc[trd2,:].values)/portfoliobasic.loc[trd2,:].values)**(1/(t3.year-t2.year)))[0]-1
dftest.loc['One RF', 'Vol'] =  np.sqrt(365/meants)*((portfolioonerandomforest.loc[t2:t3,:].values[1:]/portfolioonerandomforest.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc['Model', 'Vol'] = np.sqrt(365/meants)*((portfolio.loc[t2:t3,:].values[1:]/portfolio.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc['Model no weight', 'Vol'] = np.sqrt(365/meants)*((portfolionw.loc[t2:t3,:].values[1:]/portfolionw.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc['Model no risk management', 'Vol'] = np.sqrt(365/meants)*((portfolionr.loc[t2:t3,:].values[1:]/portfolionr.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc['Basic seasonality', 'Vol'] = np.sqrt(365/meants)*((portfoliobasic.loc[t2:t3,:].values[1:]/portfoliobasic.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc['Buy and Hold', 'Vol'] = np.sqrt(365)*((portfolioBH.loc[t2:t3,:].values[1:]/portfolioBH.loc[t2:t3,:].values[0:-1])-1).std()
dftest.loc[:, 'Sharpe Ratio'] = (dftest.loc[:,'Annualized return'])/dftest.loc[:,'Vol']
dftest.loc['Model', 'RMSE'] = np.sqrt(mean_squared_error(predictions.loc[((predictions.index>=t2) &(predictions.index<t3)), :], XP.loc[((XP.index>=t2)& (XP.index<t3)),:]))
dftest.loc['Model no risk management', 'RMSE'] = np.sqrt(mean_squared_error(predictions.loc[((predictions.index>=t2) &(predictions.index<t3)), :], XP.loc[((XP.index>=t2)& (XP.index<t3)),:]))
dftest.loc['Model no weight', 'RMSE'] = np.sqrt(mean_squared_error(predictionsnw.loc[((predictionsnw.index>=t2) &(predictionsnw.index<t3)), :], XP.loc[((XP.index>=t2)& (XP.index<t3)),:]))
dftest.loc['One RF', 'RMSE'] = np.sqrt(mean_squared_error(npy.nnPredict(rfonerandomforest,X.loc[((X.index>=pd.to_datetime(d2)) & (X.index < pd.to_datetime(d3))),colused]), 
                                                        X.loc[((X.index>=pd.to_datetime(d2)) & (X.index < pd.to_datetime(d3))),'Y']))

''' TICKS : ['FRE.DE' 'DBK.DE' 'BAS.DE' 'FME.DE'] ['HEN3.DE' 'BMW.DE' 'FME.DE' 'BAS.DE']'''
  
  
dfcv.to_csv('dfcv_nn.csv')
dftest.to_csv('dftest_nn.csv')
    
    
    
#if __name__ == '__main__':
#    main()

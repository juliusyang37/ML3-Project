"""
@author: David Abitbol, Renze Li, Feijiao Yuan, Zhihao Yang
"""

import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import os 
print( os.path.dirname(os.path.realpath(__file__)) )
def get_random_tickers(n, ticklist):
    if n >len(ticklist):
        raise Exception('n is bigger than ticklist')
    return np.random.choice(ticklist, size=n, replace=False)
RSVE = []

ticklist = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'LHA.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE']

ticklist = get_random_tickers(7, ticklist)
#ticklist = ['HEN3.DE', 'HEI.DE', 'MUV2.DE', 'CON.DE', 'DBK.DE' ,'EOAN.DE', 'BMW.DE']

print(ticklist)

#parameters : 

d0 = '2002-01-01' # begining of the waiting period
d1 = '2004-01-01' # begining of the CV period
d2 = '2005-01-01' # begining of the test period
d3 = '2005-03-01' # end of the test period
norm=True
cst=10
crossvalidationdf = pd.DataFrame(columns=['Alpha', 'Delta', 'lbd', 'Annualized return', 'Vol', 'Sharpe ratio', 'RMSE'], index = range(15))
cvind = 0
for alpha in [0.13]:
    for delta in [.91, 0.93, 0.94, 0.95]:
        for lbd in [0.3, .4, 0.5, .6]:
            
#lbd = 0.5 # history weight metric
#alpha = 0.1 # risk management metric
#cst = 10 #weight metric
#delta = 0.94

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
                                   'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Ticker',
                                   'Month', 'DAX', 'ADL', 'Type', 'Y'])
            dax = web.get_data_yahoo('^GDAXI', start=d0, end=d3)
            prices = pd.DataFrame(columns=['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose', 'Date'])
            
            # Generate the X matrix
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
                
                
                X = pd.concat([X, temp], axis=0, ignore_index=True, copy=True, sort=False)
            
            
            T = X.Ticker.unique()
            tickdict = dict(zip(T, range(len(T))))
            for tick in T:
                X.loc[X.Ticker==tick,'Ticker']= tickdict[tick]
            
            
            X.sort_values(by=['Date', 'Ticker'], inplace=True)
            X.set_index('Date', drop=True, inplace=True)
            X = X.loc[((X.index>=pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d3))), :]
            C = ['EMA10', 'EMA16', 'EMA22', 'SMA10', 'SMA16', 'SMA22', 'ValueAtRisk',
                   'Bollu20', 'Bollu26', 'Bollu32', 'Bolld20', 'Bolld26', 'Bolld32',
                   'Mom12', 'Mom18', 'Mom24', 'ACC12', 'ACC18', 'ACC24', 'ROC10', 'ROC16',
                   'ROC22', 'MACD1812', 'MACD2412', 'MACD3012', 'MACDS18129', 'MACDS24129',
                   'MACDS30129', 'RSI8', 'RSI14', 'RSI20', 'OBV', 'CHV1010', 'CHV1016',
                   'CHV1022', 'FastK12', 'FastD12', 'FastK18', 'SlowK12', 'FastD18',
                   'SlowD12', 'FastK24', 'SlowK18', 'FastD24', 'SlowD18', 'SlowK24',
                   'SlowD24', 'High', 'Low', 'Open', 'Close', 'Volume', 'AdjClose',
                   'Month', 'DAX', 'ADL','Type', 'Ticker']
            Xtrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), C]
            Xcv1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), C]
            Ytrain1 = X.loc[((X.index>pd.to_datetime(d0)) & (X.index<=pd.to_datetime(d1))), 'Y']
            Ycv1 = X.loc[((X.index>pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d2))), 'Y']
            

                
            ### Cross Validation for features:
            #order features:
            nfeatures=len(Xtrain1.columns)
            rmse = []
            colsrmse =[]
            rf = RandomForestRegressor(n_estimators=100, max_features=None).fit(Xtrain1, Ytrain1)
            df2 = pd.DataFrame(index=Xtrain1.columns, columns=['rankval'])
            df2.rankval = rf.feature_importances_
            df2.sort_values(by='rankval', inplace=True, ascending=False)
            yhat = rf.predict(Xcv1)
            fig = plt.figure(figsize=(10,15))
            plt.title('Variables relative importance')
            plt.barh(np.arange(len(df2.index)),
                     df2.rankval, align='center')
            plt.yticks(range(len(df2.index)), df2.index.values.tolist())
            plt.xlabel('Relative Importance')
            plt.show()
            rmse.append(np.sqrt(mean_squared_error(yhat, Ycv1)))
            colsrmse.append(df2.index)
            CL = []
            CL.append(df2.index.values)
            cols = df2.index.values[df2.index!=df2.index[-1]]
            
            for i in range(1,len(df2.index)):
                Xtemp = Xtrain1.loc[:,cols]
                CL.append(cols)
                rf = RandomForestRegressor(n_estimators=100, max_features=None).fit(Xtemp, Ytrain1)
                df2 = pd.DataFrame(index=Xtemp.columns, columns=['rankval'])
                df2.rankval = rf.feature_importances_
                df2.sort_values(by='rankval', inplace=True, ascending=False)
                yhat = rf.predict(Xcv1.loc[:,cols])
                mse = mean_squared_error(yhat, Ycv1)
                rmse.append(np.sqrt(mse))
                colsrmse.append(df2.index) 
                cols = df2.index.values[df2.index!=df2.index[-1]]
                
            plt.scatter(range(len(rmse)), np.array(rmse[::-1]))
            plt.title('RMSE versus number of features')
            plt.xlabel('Number of features')
            plt.ylabel('RMSE')
            plt.show()
            #Gives features we use
            colused = CL[::-1][np.argmin(rmse[::-1][10:30])+10]
            
            
            ## Train
            #X.sort_values(by=['Date','Ticker'], inplace=True)
            
            
            rfL = []
            dtlist = X.index.unique().sort_values()
            first_day= dtlist[0]
            d = first_day+dt.timedelta(50)
            while d < pd.to_datetime(d3):
                X50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),colused]
                Y50_train = X.loc[((X.index>=d+dt.timedelta(-50)) & (X.index<d)),'Y']
                rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X50_train, Y50_train)
                rfL.append((d, rf))
                if d >= first_day + dt.timedelta(100):
                    X100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),colused]
                    Y100_train = X.loc[((X.index>=d+dt.timedelta(-100)) & (X.index<d)),'Y']
                    rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X100_train, Y100_train)
                    rfL.append((d,rf))
                if d >= first_day + dt.timedelta(200):
                    X200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),colused]
                    Y200_train = X.loc[((X.index>=d+dt.timedelta(-200)) & (X.index<d)),'Y']
                    rf = RandomForestRegressor(n_estimators=200, max_features=None).fit(X200_train, Y200_train)
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
            portfolio = pd.DataFrame(columns=['P'], index=X.loc[((X.index>=pd.to_datetime(d1)) & (X.index<=pd.to_datetime(d3))),:].index.unique().sort_values())
            
            
            
            j=0
            while d < pd.to_datetime(d3):
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
                    rp = rfL[0][1].predict(X.loc[d,feats]) ##
                    rmse[0] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y'])) ##
                    if w[0] != 0 : w[0] = calc_wi(k[0], deltaT)
                    else: 
                        
                        w[0] = 1
                        k[0] = 1/rmse[0]
                    
                    BS =(rp / np.abs(rp).sum())
                    
                    
                except:
                    d=pd.to_datetime(d)
                    k[0] = calc_ki(lbd, k[0], rmse[0])
                    rp = rfL[0][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                    rmse[0] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']])) ##
                    if w[0] != 0 : w[0] = calc_wi(k[0], deltaT)
                    else: 
                        
                        w[0] = 1
                        k[0] = 1/rmse[0]
                    
                    
                try:
                    temp = np.zeros(ntick)
                    temp2 = rfL[0][1].predict(X.loc[d,feats].values)
                    for tick in X.loc[d,'Ticker'].sort_values():
                        temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
                except:
                    temp2 = rfL[0][1].predict(X.loc[d,feats].values.reshape(1, -1))
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
                            
                            rp = rfL[i][1].predict(X.loc[d,feats])
                            BS =(rp / np.abs(rp).sum())
                            
                            rmse[i] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
                        except:
                            d=pd.to_datetime(d)
                            
                            k[i] = calc_ki(lbd, k[i], rmse[i])
                            
                            w[i] = calc_wi(k[i], deltaT)
                            
                            rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                            
                            rmse[i] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
                    else:
                        try:
                            l = len(X.loc[d,feats].columns)
                            d=pd.to_datetime(d)
                            mn[i] = w[:nrfprev].mean()
                            w[i] = w[:nrfprev].mean()
                            rp = rfL[i][1].predict(X.loc[d,feats])
                            BS =(rp / np.abs(rp).sum())
                            
                            rmse[i] = np.sqrt(mean_squared_error(rp, X.loc[d,'Y']))
                            k[i] = 1/rmse[i]
                        except:
                            d=pd.to_datetime(d)
                            mn[i] = w[:nrfprev].mean()
                            w[i] = w[:nrfprev].mean()
                            rp = rfL[i][1].predict(pd.DataFrame(X.loc[d,feats].values.reshape(1,-1)))
                            
                            rmse[i] = np.sqrt(mean_squared_error(rp, [X.loc[d,'Y']]))
                            k[i] = 1/rmse[i]
                    RSVE.append(rmse[i])
                    try:
                        temp = np.zeros(ntick)
                        temp2 = rfL[i][1].predict(X.loc[d,feats])
                        for tick in X.loc[d,'Ticker'].sort_values():
                            temp[tick] = temp2[X.loc[d,'Ticker'].values == tick].sum()
                    except:
                        temp2 = rfL[i][1].predict(X.loc[d,feats].values.reshape(1, -1))
                        temp = np.zeros(ntick)
                        temp[X.loc[d,'Ticker']] = temp2
                    P.append(temp)
                    
                if d >= pd.to_datetime(d1): 
                    
                    P = pd.DataFrame(P)
                    predictions.loc[d, :] = (P.values * np.array([w[:nrf].tolist()]*ntick).transpose()).sum(axis=0)/w.sum()
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
                    else : 
                        c = portfolio.iloc[j-1,0]                      
                    wd = predictions.loc[d, :].values/predictions.loc[d, :].abs().sum()
                    ret = np.zeros(ntick)
                    try:
                        X.loc[d,:].Ticker.unique()
                        for tick in X.loc[d,:].Ticker.unique():
                            ret[tick] = X.loc[((X.index==d) & (X.Ticker==tick)),'Y'].values[0]/100
                    except: 
                        ret[X.loc[d,:].Ticker] = X.loc[d,:].Y/100
                    profit = (wd * c * ret * D).sum()
                    portfolio.iloc[j, 0] = c+profit
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
                          
                        
            xa=range(len(w0))
            plt.figure(figsize=(15,15))
        
            plt.plot(xa, w1, label='2')
           
            plt.ylim(7,10)
            plt.plot(xa, w9, label='10')
            
            plt.legend()
            plt.show()
            
            
                
            retholdcv = np.zeros(ntick)
            retholdtest = np.zeros(ntick)
            prices.drop_duplicates(subset=['Date', 'Ticker'])
            prices2 = prices.set_index('Date', drop=True)
            stocks = pd.DataFrame(index = prices.Date.sort_values().unique(), columns=tickdict.keys())
            XP = pd.DataFrame(index = predictions.index, columns=predictions.columns)
            X2 = X.reset_index()
            X2 = X2.drop_duplicates(['Date', 'Ticker'])
            X2 = X2.set_index('Date')
            for tick in tickdict.keys():
                prices.Date = pd.to_datetime(prices.Date)
                prices.loc[prices.Ticker==tick,'Ticker'] = tickdict[tick]
                t2 = tickdict[tick]
                p1 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Close'].values[0]
                p2 = prices.loc[((prices.Date>=pd.to_datetime(d1)) & (prices.Date<pd.to_datetime(d2)) & (prices.Ticker==t2)), 'Open'].values[-1]
                p3 = prices.loc[((prices.Date>=pd.to_datetime(d2)) & (prices.Date<pd.to_datetime(d3)) & (prices.Ticker==t2)), 'Close'].values[0]
                p4 = prices.loc[((prices.Date>=pd.to_datetime(d2)) & (prices.Date<pd.to_datetime(d3)) & (prices.Ticker==t2)), 'Open'].values[-1]
                retholdcv[t2] = p2/p1-1
                retholdtest[t2] = p4/p3-1
                stocks.loc[:,tick] = prices2.loc[prices2.Ticker==tick,:].loc[stocks.index,'Close']
                temp = X2.loc[XP.index, :]
                XP.loc[:,t2] = temp.loc[temp.Ticker==t2,'Y'].values
            stocksreturn = pd.DataFrame(stocks.iloc[1:,:].values / stocks.iloc[0:-1,:].values, index=stocks.index[1:])-1
            rs = (1 + stocksreturn.mean(axis=1)).values
            
            
            holdcv = retholdcv.mean() 
            holdtest = retholdtest.mean()   
            
            

            t0 = pd.to_datetime(d0)
            t1 = pd.to_datetime(d1)
            t2 = pd.to_datetime(d2)
            t3 = pd.to_datetime(d3)
            trd0 = dtlist[dtlist>=t0][0]
            trd1 = dtlist[dtlist>=t1][0]
            trd2 = dtlist[dtlist>=t2][0]
            trd3 = dtlist[dtlist>=t2][-1]
            meants = np.mean((dtlist[1:]-dtlist[0:-1]).days)
            daxrcv = (dax.loc[dax.index[dax.index>=t1][0], 'Close']/dax.loc[dax.index[dax.index<t2][-1], 'Close'])**(1/(t2.year-t1.year))-1
            crossvalidationdf.loc[cvind, 'Annualized return'] = ((1+(portfolio.loc[trd2,:].values - 1)**(1/(t2.year-t1.year))))[0]-1
            crossvalidationdf.loc[cvind, 'Vol'] = np.sqrt(365/meants)*((portfolio.loc[t1:t2,:].values[1:]/portfolio.loc[t1:t2,:].values[0:-1])-1).std()
            crossvalidationdf.loc[cvind, 'Sharpe ratio'] = (crossvalidationdf.loc[cvind,'Annualized return'] - daxrcv)/crossvalidationdf.loc[cvind, 'Vol']
            crossvalidationdf.loc[cvind, 'RMSE'] = np.sqrt(mean_squared_error(predictions.loc[((predictions.index>=t1) &(predictions.index<t2)), :], XP.loc[((XP.index>=t1)& (XP.index<t2)),:]))
            crossvalidationdf.loc[cvind, 'Alpha'] = alpha
            crossvalidationdf.loc[cvind, 'Delta'] = delta
            crossvalidationdf.loc[cvind, 'lbd'] = lbd
            
            ''' TICKS : ['FRE.DE' 'DBK.DE' 'BAS.DE' 'FME.DE'] ['HEN3.DE' 'BMW.DE' 'FME.DE' 'BAS.DE']'''
                
            crossvalidationdf.to_csv('crossvalidation_table.csv')
            cvind+=1
            print(crossvalidationdf)
    
    
    

    
    
    
#if __name__ == '__main__':
#    main()

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# Benchmark to track
BENCHMARK = '^NSEI'

startDateA = '2009-01-01'
endDateA = '2022-09-30'

EQUITIES = {1: ['ADANIENT','HINDALCO','TATASTEEL','SAIL'], #METALS
            2: ['WIPRO','HCLTECH','INFY','TCS'], #IT
            3: ['DLF','UNITECH'], #REALTY
            4: ['RCOM','BHARTIARTL','IDEA'], #TELECOM
            5: ['ABB','SIEMENS','BHEL'], #ENGINEERING
            6: ['ACC','AMBUJACEM','ULTRACEMCO'], #CEMENT
            7: ['CIPLA','AUROPHARMA','SUNPHARMA'], #PHARMA
            8: ['HDFC','RELCAPITAL','IDFC'], #FINSERV
            9: ['HEROMOTOCO','M&M','MARUTI','TATAMOTORS'], #AUTO
            10: ['ITC','HINDUNILVR','MCDOWELL-N'], #CONSUMER
            11: ['KOTAKBANK','ICICIBANK','HDFCBANK','AXISBANK','PNB','SBIN'], #BANKING
            12:['NTPC','RPOWER','SUZLON','TATAPOWER','POWERGRID'], #ENERGY-POWER
            13:['GAIL','RELIANCE','ONGC','BPCL'], #ENERGY - OIL & GAS
            14:['RELINFRA','JPASSOCIAT','LT'] #CONSTRUCTION
            }

# Number of available tickers in each Sector
SECTOR_SIZE = {1: 4, 2: 4, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 4, 10: 3, 11: 6, 12: 5, 13:4 , 14:3}

# Minimum and Maximum weight for each Sector within Portfolio
SECTOR_WEIGHTS = {1: (0.0, 0.30), 2: (0.0, 0.30), 3: (0.0, 0.30), 4: (0.0, 0.30), 5: (0.0, 0.30),
                  6: (0.0, 0.30), 7: (0.0, 0.30), 8: (0.0, 0.30), 9: (0.0, 0.30), 10: (0.0, 0.30),
                  11: (0.0, 0.30),12: (0.0, 0.30),13: (0.0, 0.30),14: (0.0, 0.30)}


# Months to look back while computing weights
LOOKBACK_MONTHS = 12

#Max weight of a stock
MAX_EQ_WT = 0.05

#Monthly rebalancing limit
MONTHLY_CHANGE = 0.05

#Transaction fees
TRANSACTION_FEES = 0.01


class PortfolioOptimization():

    def __init__(self):
        # Creating list of all Equity tickers
        self.Equities = [item for sublist in [EQUITIES[x] for x in EQUITIES.keys()] for item in sublist]
        # Flag for Optimizer to disable auto-correlation for first iteration
        self.FirstIter = True
        self.OldWeights = []
        self.NewWeights = []

        self.P = []
        self.B = []
        self.D = []

        #To save monthly returns
        self.SPs = []
        self.SPs2 = []
        self.Iter = 1


    # To download Equity and Benchmark data
    def Download_data(self, startDate, endDate, overwrite=False):

        # Creating Data folder if it does not exists
        if not os.path.exists('DATA'):
            print("'DATA' directory not found!\nCreating 'DATA' directory")
            os.mkdir('DATA')

        # Downloading Equity and Benchmark data
        for ticker in self.Equities + [BENCHMARK]:
            if (overwrite == True) or (overwrite == False and not os.path.exists('DATA/' + ticker + '.csv')):
                print('Downloading data for ' + ticker)
                if ticker!='^NSEI' :
                    df = yf.download(ticker+'.NS', start=startDate, end=endDate)
                else:
                    df = yf.download(ticker, start=startDate, end=endDate)

                print(len(df))

                startDateA = startDate.split('-')
                startDateA = startDateA[1] + '-' + startDateA[2] + '-' + startDateA[0]

                endDateA = endDate.split('-')
                endDateA = endDateA[1] + '-' + endDateA[2] + '-' + endDateA[0]

                # Filling out missing dates and values
                idx = pd.date_range(startDateA, endDateA)
                df = df.reindex(idx, fill_value=np.nan)
                df = df.ffill()
                df = df.bfill()

                # Saving data
                df.to_csv('DATA/' + ticker + '.csv')



    # To load data and converting to percentage change
    def Load_data(self):
        arr = []
        # retrieving Equities as a matrix
        for ticker in self.Equities:
            df = pd.read_csv('DATA/' + ticker + '.csv', index_col=[0])

            df = df.pct_change()
            df = df.iloc[1:, ]
            arr += [df['Close'].values]

        arr = np.transpose(np.array(arr))

        # Retrieving benchmark
        df = pd.read_csv('DATA/' + BENCHMARK + '.csv', index_col=[0])
        df = df.pct_change()
        df = df.iloc[1:, ]

        bench_df = df.copy()

        # List of dates
        dates = df.index.tolist()

        port_df = pd.DataFrame(np.hstack(([[a] for a in dates], arr)), columns=['Date'] + self.Equities)
        port_df['Date'] = pd.to_datetime(port_df['Date'])
        port_df_splits = [group for _, group in port_df.groupby(pd.Grouper(key='Date', freq='M'))]

        bench_df.index.name = "Date"
        bench_df = bench_df[['Close']]
        bench_df.reset_index(inplace=True)
        bench_df['Date'] = pd.to_datetime(bench_df['Date'])
        bench_df_splits = [group for _, group in bench_df.groupby(pd.Grouper(key='Date', freq='M'))]

        # return arr, bench, dates

        return port_df_splits, bench_df_splits



    def Opt_SR(self, p, b):
        p = np.array(p.values).astype(np.float)
        b = np.array(b.values).astype(np.float)

        #to form sectors for sector constraints
        SecLens = {}
        SecLens[1] = (0, SECTOR_SIZE[1] - 1)
        for i in range(2, max(SECTOR_SIZE.keys()) + 1):
            SecLens[i] = (SecLens[i - 1][1] + 1, SecLens[i - 1][1] + SECTOR_SIZE[i])

        #set initial weights
        w0 = np.array([0.1] * (SecLens[max(SecLens.keys())][1] + 1))

        if self.FirstIter == False:
            w0 = self.OldWeights


        bnds = tuple()
        bnds += tuple([tuple(x) for x in [[0.0, MAX_EQ_WT]] * len(w0)])

        #Sortino ratio optimization function
        def opt(w):
            return -(np.mean(np.sum(p * w, axis=1)) / np.std(np.sum(  ((p * w)-np.abs(p*w))/2  , axis=1)))

        #Sector weights constraints
        def sum_con(w):
            return np.sum(w) - 1

        def S1_min(w):
            return np.sum(w[SecLens[1][0]:SecLens[1][1] + 1]) - SECTOR_WEIGHTS[1][0]

        def S1_max(w):
            return -np.sum(w[SecLens[1][0]:SecLens[1][1] + 1]) + SECTOR_WEIGHTS[1][1]

        def S2_min(w):
            return np.sum(w[SecLens[2][0]:SecLens[2][1] + 1]) - SECTOR_WEIGHTS[2][0]

        def S2_max(w):
            return -np.sum(w[SecLens[2][0]:SecLens[2][1] + 1]) + SECTOR_WEIGHTS[2][1]

        def S3_min(w):
            return np.sum(w[SecLens[3][0]:SecLens[3][1] + 1]) - SECTOR_WEIGHTS[3][0]

        def S3_max(w):
            return -np.sum(w[SecLens[3][0]:SecLens[3][1] + 1]) + SECTOR_WEIGHTS[3][1]

        def S4_min(w):
            return np.sum(w[SecLens[4][0]:SecLens[4][1] + 1]) - SECTOR_WEIGHTS[4][0]

        def S4_max(w):
            return -np.sum(w[SecLens[4][0]:SecLens[4][1] + 1]) + SECTOR_WEIGHTS[4][1]

        def S5_min(w):
            return np.sum(w[SecLens[5][0]:SecLens[5][1] + 1]) - SECTOR_WEIGHTS[5][0]

        def S5_max(w):
            return -np.sum(w[SecLens[5][0]:SecLens[5][1] + 1]) + SECTOR_WEIGHTS[5][1]

        def S6_min(w):
            return np.sum(w[SecLens[6][0]:SecLens[6][1] + 1]) - SECTOR_WEIGHTS[6][0]

        def S6_max(w):
            return -np.sum(w[SecLens[6][0]:SecLens[6][1] + 1]) + SECTOR_WEIGHTS[6][1]

        def S7_min(w):
            return np.sum(w[SecLens[7][0]:SecLens[7][1] + 1]) - SECTOR_WEIGHTS[7][0]

        def S7_max(w):
            return -np.sum(w[SecLens[7][0]:SecLens[7][1] + 1]) + SECTOR_WEIGHTS[7][1]

        def S8_min(w):
            return np.sum(w[SecLens[8][0]:SecLens[8][1] + 1]) - SECTOR_WEIGHTS[8][0]

        def S8_max(w):
            return -np.sum(w[SecLens[8][0]:SecLens[8][1] + 1]) + SECTOR_WEIGHTS[8][1]

        def S9_min(w):
            return np.sum(w[SecLens[9][0]:SecLens[9][1] + 1]) - SECTOR_WEIGHTS[9][0]

        def S9_max(w):
            return -np.sum(w[SecLens[9][0]:SecLens[9][1] + 1]) + SECTOR_WEIGHTS[9][1]

        def S10_min(w):
            return np.sum(w[SecLens[10][0]:SecLens[10][1] + 1]) - SECTOR_WEIGHTS[10][0]

        def S10_max(w):
            return -np.sum(w[SecLens[10][0]:SecLens[10][1] + 1]) + SECTOR_WEIGHTS[10][1]

        def S11_min(w):
            return np.sum(w[SecLens[11][0]:SecLens[11][1] + 1]) - SECTOR_WEIGHTS[11][0]

        def S11_max(w):
            return -np.sum(w[SecLens[11][0]:SecLens[11][1] + 1]) + SECTOR_WEIGHTS[11][1]

        cons = [{'type': 'eq', 'fun': sum_con},
                {'type': 'ineq', 'fun': S1_min},
                {'type': 'ineq', 'fun': S1_max},
                {'type': 'ineq', 'fun': S2_min},
                {'type': 'ineq', 'fun': S2_max},
                {'type': 'ineq', 'fun': S3_min},
                {'type': 'ineq', 'fun': S3_max},
                {'type': 'ineq', 'fun': S4_min},
                {'type': 'ineq', 'fun': S4_max},
                {'type': 'ineq', 'fun': S5_min},
                {'type': 'ineq', 'fun': S5_max},
                {'type': 'ineq', 'fun': S6_min},
                {'type': 'ineq', 'fun': S6_max},
                {'type': 'ineq', 'fun': S7_min},
                {'type': 'ineq', 'fun': S7_max},
                {'type': 'ineq', 'fun': S8_min},
                {'type': 'ineq', 'fun': S8_max},
                {'type': 'ineq', 'fun': S9_min},
                {'type': 'ineq', 'fun': S9_max},
                {'type': 'ineq', 'fun': S10_min},
                {'type': 'ineq', 'fun': S10_max},
                {'type': 'ineq', 'fun': S11_min},
                {'type': 'ineq', 'fun': S11_max}]

        #Add monthly change/turnover constraint if no first itertion
        def MC(w):
            return -(np.sum(np.abs(self.OldWeights - w)) / 2) + MONTHLY_CHANGE

        if self.FirstIter == False:
            cons += [{'type': 'ineq', 'fun': MC}]
        res = minimize(opt, w0, constraints=cons, bounds=bnds, method='slsqp',
                        options={'xatol': 1e-8, 'disp': True, 'maxiter': 1000})

        return res.x



    #to simulate, plot and compute monthly returns
    def Sim(self, w, p, b):
        d = p['Date']
        p = np.array(p.drop('Date', axis=1).values).astype(np.float) + 1
        b = np.array(b.drop('Date', axis=1).values).astype(np.float) + 1

        print('IR:')
        IR = (np.mean((np.subtract(np.sum(p * w, axis=1), b))) / np.std(np.subtract(np.sum(p * w, axis=1), b)))*math.sqrt(252)
        print(IR)


        #add transaction fees if not first iteration
        if(self.FirstIter==True):
            p[0] = np.multiply(w, p[0] )
        else:
            p[0] = np.multiply(w, (p[0]- (np.abs(self.OldWeights-w)*TRANSACTION_FEES)) )

        #run simulation
        for i in range(1, len(p)):
            p[i] = np.multiply(p[i - 1], p[i])
            b[i] = np.multiply(b[i - 1], b[i])

        ow = (p[-1]) / np.sum(p[-1])
        p = np.sum(p, axis=1)

        #save portfolio and benchmark monthly returns
        self.SPs += [(p[-1]-1 )]
        self.SPs2 += [(b[-1]-1)]

        #plot and save monthly return charts
        plt.plot(pd.to_datetime(d, format='%Y-%m-%d'), p, color='r')
        plt.plot(pd.to_datetime(d, format='%Y-%m-%d'), b, color='b')
        plt.savefig('RESULTS/Fig-' + str(self.Iter) + '.png')
        self.Iter += 1
        plt.clf()

        #save simulations for overall simulation chart
        if self.FirstIter == True:
            self.P = p
            self.B = b
            self.D = d
        else:
            self.P = np.concatenate((self.P, p * self.P[-1]), axis=0)
            self.B = np.concatenate((self.B, b * self.B[-1]), axis=0)
            self.D = np.concatenate((self.D, d), axis=0)

        #update old weights
        self.OldWeights = ow




    def run(self):
        # load stocks and benchmark data
        pdf, bdf = self.Load_data()

        #create directory to save monthly return charts
        if not os.path.exists('RESULTS'):
            print("'RESULTS' directory not found!\nCreating 'RESULTS' directory")
            os.mkdir('RESULTS')

        #slice past 12 months data for optimizer to calculate weights
        for month in tqdm(range(LOOKBACK_MONTHS, len(pdf))):
            pdf_ = pd.concat([pdf[x] for x in range(month - LOOKBACK_MONTHS, month )],
                             ignore_index=True, axis=0)
            bdf_ = pd.concat([bdf[x] for x in range(month - LOOKBACK_MONTHS, month )],
                             ignore_index=True, axis=0)

            pdf_ = pdf_.drop('Date', axis=1)
            bdf_ = bdf_.drop('Date', axis=1)
            pdf_.dropna(axis=0, how='all', inplace=True)
            bdf_.dropna(axis=0, how='all', inplace=True)

            #optimizer function call
            self.NewWeights=self.Opt_SR(pdf_, bdf_)

            #call Sim function
            self.Sim(self.NewWeights, pdf[month], bdf[month])
            self.FirstIter = False

        #save monthly returns
        self.SPs=np.array(self.SPs)
        self.SPs2 = np.array(self.SPs2)
        SPs_down = self.SPs[self.SPs<0]
        SPs2_down = self.SPs2[self.SPs2 < 0]

        out_df = pd.concat( [pdf[x] for x in range( LOOKBACK_MONTHS, len(pdf))]   ,
                             ignore_index=True, axis=0)
        out_df['month_year'] = out_df['Date'].dt.to_period('M')
        dates = out_df['month_year'].unique()

        dff = pd.DataFrame()
        dff['MONTH']=dates
        dff['PORTFOLIO']=self.SPs
        dff['BENCHMARK']=self.SPs2

        dff.to_csv('Returns.csv',index=False)

        #print Sortino and IR
        print('---------')
        print('IR')
        print((np.mean(self.SPs-self.SPs2) / np.std(self.SPs-self.SPs2)) * math.sqrt(12))
        print('SORTINO')
        print("PORTFOLIO: ", end=' ')
        print((np.mean(self.SPs) / np.std(SPs_down)) * math.sqrt(12))
        print("BENCHMARK: ", end=' ')
        print((np.mean(self.SPs2) / np.std(SPs2_down)) * math.sqrt(12))
        plt.plot(pd.to_datetime(self.D, format='%Y-%m-%d'), self.P, color='r')
        plt.plot(pd.to_datetime(self.D, format='%Y-%m-%d'), self.B, color='b')
        plt.plot()
        plt.savefig('Returns.png')
        plt.show()


if __name__ == "__main__":
    POPT = PortfolioOptimization()
    POPT.Download_data(startDate=startDateA, endDate=endDateA, overwrite=False)
    POPT.run()








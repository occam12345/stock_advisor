import FinanceDataReader as fdr
import requests
import bs4
import datetime
import warnings
warnings.filterwarnings('ignore')
import json
from IPython.display import clear_output
import glob
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle

with open('rf', 'rb') as file:
    rf = pickle.load(file)

# Get the data from Nave Chart
def make_price_data(code, timeframe, count):
    url = 'https://fchart.stock.naver.com/sise.nhn?symbol=' + code + '&timeframe=' + timeframe + '&count=' + count + '&requestType=0'
    price_data = requests.get(url)
    price_data_bs = bs4.BeautifulSoup(price_data.text, 'lxml')
    item_list = price_data_bs.find_all('item')

    date_list = []
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    trade_list = []

    for item in item_list:
        data = item['data'].split('|')
        date_list.append(data[0])
        open_list.append(data[1])
        high_list.append(data[2])
        low_list.append(data[3])
        close_list.append(data[4])
        trade_list.append(data[5])

    price_df = pd.DataFrame(
        {'open': open_list, 'high': high_list, 'low': low_list, 'close': close_list, 'volume': trade_list},
        index=date_list)
    price_df['code'] = code
    num_vars = ['open', 'high', 'low', 'close', 'volume']
    char_vars = ['code']
    price_df = price_df.reindex(columns=char_vars + num_vars)

    for var in num_vars:
        price_df[var] = pd.to_numeric(price_df[var], errors='coerce')

    price_df.index = pd.to_datetime(price_df.index, errors='coerce')
    return price_df

def kosdaq_index():
    kosdaq_index = fdr.DataReader('KQ11', '2000')
    kosdaq_index.columns = ['close', 'open', 'high', 'low', 'volume', 'change']
    kosdaq_index.index.name = 'date'
    kosdaq_index.sort_index(inplace=True)
    kosdaq_index['kosdaq_return'] = kosdaq_index['close'] / kosdaq_index['close'].shift(1)
    # print(kosdaq_index.index.max())
    return kosdaq_index

def strategy_implement(return_all, kosdaq_list, decision_date):

    start_dt = decision_date
    end_dt = decision_date

    w_days = list(pd.bdate_range(start_dt, end_dt)) # 백테스트할 날짜모음
    for dt in w_days: # 날짜를 하루 하루 증가시키면서 시뮬레이션

        obs_end = datetime.datetime.strftime(dt, "%Y-%m-%d")
        start_date = datetime.datetime.strptime(obs_end, "%Y-%m-%d") - datetime.timedelta(days=180) # 총 180 일 관찰기간
        obs_start = datetime.datetime.strftime(start_date, "%Y-%m-%d")

        code_list = []
        name_list = []
        price_list = []
        price_mean_list = []
        price_std_list = []
        price_z_list = []
        price_z_out_list = []
        price_5ma_list = []
        price_low_list = []
        price_high_list = []
        volume_list = []
        volume_mean_list = []
        volume_std_list = []
        volume_z_list = []
        volume_z_out_list = []
        wins_60_list = []
        wins_180_list = []
        toptail_list = []

        for code, name in zip(kosdaq_list['code'], kosdaq_list['name']):
            s = return_all[return_all['code']==code].loc[obs_start:obs_end]

            if (s['volume'].sum() < 1000000) or (s['close'].std() == 0) or (s['low'].min() == 0): # 6 개월간 총 거래량이 백만은 넘겠지...
                continue

            code_list.append(code)
            name_list.append(name)

            price_list.append(s['close'].tail(1).max()) # 마지막날 종가
            price_mean_list.append(s['close'].mean()) # 종가 평균
            price_std_list.append(s['close'].std()) # 종가 표준편차
            price_z =  (s['close'].tail(1).max() - s['close'].mean()) / s['close'].std() # 어제 Z 값
            price_z_out =  np.where( ((s['close'] - s['close'].mean()) / s['close'].std()) > 1.6, 1, 0) # 특이하게 높은 값 기록
            price_5ma_list.append(s['close'].tail(5).mean())# 5일 이동평균선
            price_low_list.append(s['low'].tail(1).max())  # 마지막날 저가
            price_high_list.append(s['high'].tail(1).max())  # 마지막날 고가

            volume_list.append(s['volume'].tail(1).max()) # 마지막날 거래량
            volume_mean_list.append(s['volume'].mean()) # 거래량 평균
            volume_std_list.append(s['volume'].std()) # 거래량 표준편차
            volume_z = (s['volume'].tail(1).max() - s['volume'].mean()) / s['volume'].std()  # 어제 Z 값
            volume_z_out = np.where( ((s['volume'] - s['volume'].mean()) / s['volume'].std()) > 1.6, 1, 0) # 특이하게 거래량 높은 날 기록

            price_z_list.append(price_z)
            volume_z_list.append(volume_z)

            price_z_out_list.append(price_z_out.sum())
            volume_z_out_list.append(volume_z_out.sum())

            wins_60_list.append(s['win_market'].tail(60).sum()) # 지난 180일 동안 코스닥 인덱스보다 잘 한 날짜 수
            wins_180_list.append(s['win_market'].sum()) # 지난 60일 동안 코스닥 인덱스보다 잘 한 날짜 수

            toptail = (s['close'] > s['open'])*((s['high']/s['close'])>1.05).astype('int') # 양봉이고 5% 이상 위 꼬리 상승
            toptail_list.append(toptail.sum())

        # 데이터프레임으로 전환 - 종목레벨 데이터
        wins = pd.DataFrame({'code': code_list,'name': name_list, 'price': price_list, 'price_mean': price_mean_list, 'price_std': price_std_list, 'price_z': price_z_list,'price_z_out': price_z_out_list,\
                             'price_5ma': price_5ma_list, 'price_high': price_high_list, 'price_low': price_low_list,\
                             'volume': volume_list, 'volume_mean': volume_mean_list, 'volume_std': volume_std_list, 'volume_z': volume_z_list, 'volume_z_out': volume_z_out_list,\
                             'num_wins_60': wins_60_list, 'num_wins_180': wins_180_list, 'num_toptail': toptail_list})


        wins['buy_price'] = wins['price']*0.985 # 매수 가격은 마지막날 종가에서 1.5% 빠진 가격
        wins['price_z'] = (wins['price'] - wins['price_mean'])/wins['price_std'] # 마지막날 가격의 Z 스코어
        wins['volume_z'] = (wins['volume'] - wins['volume_mean'])/wins['volume_std'] # 마지막날 거래량의 Z 스코어

        c1 = (wins['num_wins_180'] > 20)  # 지난 180일 동안 코스닥 인덱스보다 잘 한 날짜 수 > 20 (한달 영영일)
        c2 = (wins['num_wins_60'] > 10)
        candidates = wins[ c1 & c2 ] # 위 조건을 만족

        outcome_all = pd.DataFrame()

        for index, row in candidates.iterrows():

            outcome = return_all[return_all['code']==row['code']].loc[obs_end : obs_end][['code', 'low', 'close']] # 필요한 컬럼만
            outcome['name'] = row['name']
            outcome['buy_price'] = int(row['buy_price']) # 어제 종가
            outcome['price_z'] = row['price_z'] # 어제 정보
            outcome['volume_z'] = row['volume_z'] # 어제 정보
            outcome['price_z_out'] = row['price_z_out'] # 180일 요약 정보
            outcome['price_5ma'] = row['price_5ma'] # 5일 요약 정보
            outcome['volume_z_out'] = row['volume_z_out'] # 180일 요약 정보
            outcome['num_wins_60'] = row['num_wins_60'] # 60일 요약 정보
            outcome['num_wins_180'] = row['num_wins_180'] # 180일 요약 어제 정보
            outcome['num_toptail'] = row['num_toptail'] # 180일 요약 어제 정보
            outcome['num_wins_trend'] = outcome['num_wins_60']/outcome['num_wins_180']
            outcome['price_from_5ma'] = outcome['buy_price']/outcome['price_5ma']
            outcome['price_high'] = row['price_high'] # 어제 정보
            outcome['price_low'] = row['price_low'] # 어제 정보
            outcome['price_change'] = outcome['price_high']/outcome['price_low']  # 어제 정보
            outcome['yymmdd'] = obs_end

            indep = ['price_change', 'price_z', 'volume_z', 'price_z_out', 'volume_z_out', 'num_wins_60', 'num_wins_trend', 'price_from_5ma', 'num_toptail']

            X = outcome[indep]
            outcome['yhat'] = rf.predict_proba(X)[:,1]
            outcome['select'] = np.where(outcome['yhat']  > 0.55, 1, 0)
            outcome_all = pd.concat([outcome_all, outcome], axis=0)

    return outcome_all


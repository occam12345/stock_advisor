
import kq_stocks as kq
import pandas as pd
import numpy as np
import streamlit as st

# decision_date = input("오늘 정보를 기반으로 내일 매수할 종목을 추천합니다. 오늘 날짜를 YYYY-MM-DD 형태로 입력하세요: ")
kosdaq_list = pd.read_pickle('kosdaq_code.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    st.title('코스닥 주식 어드바이저')
    decision_date = st.text_input("오늘 날짜를 다음과 같은 포맷으로 입력하세요. 포맷:  YYYY-MM-DD")
    if decision_date:

        price_data = pd.DataFrame()
        for code in kosdaq_list['code']:
            daily_price = kq.make_price_data(code, 'day', '300')
            price_data = pd.concat([price_data, daily_price], axis=0)
        price_data.index.name = 'date'

        kosdaq_index = kq.kosdaq_index()    #
        price_kosdaq = price_data.merge(kosdaq_index['kosdaq_return'], left_index=True, right_index=True, how='left')  # merge individual stock price data with KOSPI index

        # 인덱스 값과 비교하여 인덱스보다 잘 하난 날 카운트
        return_all = pd.DataFrame()
        for code in kosdaq_list['code']:
            stock_return = price_kosdaq[price_kosdaq['code'] == code].sort_index()
            stock_return['return'] = stock_return['close'] / stock_return['close'].shift(1)
            c1 = (stock_return['kosdaq_return'] < 1)
            c2 = (stock_return['return'] > 1)
            stock_return['win_market'] = np.where((c1 & c2), 1, 0)
            return_all = pd.concat([return_all, stock_return], axis=0)

        # decision_date2 = "'" + decision_date + "'"

        outcome_all = kq.strategy_implement(return_all, kosdaq_list, decision_date)
        kq_selection = outcome_all[outcome_all['select'] == 1][['code', 'name', 'buy_price', 'yhat']]

        kq_selection.to_pickle('kq_selection.pkl')
        kq_selection = pd.read_pickle('kq_selection.pkl')
        st.write(kq_selection.sort_values(by='yhat', ascending=False))




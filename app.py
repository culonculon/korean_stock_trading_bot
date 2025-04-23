import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from stock_data import StockData
from technical_indicators import TechnicalIndicators
from trading_signals import TradingSignals
from config import STOCK_CODES

# 다중 인덱스 처리를 위한 공통 헬퍼 함수
def has_column(df, col_name):
    """데이터프레임에 컬럼이 존재하는지 확인(다중 인덱스 지원)"""
    if isinstance(df.columns, pd.MultiIndex):
        return any(col[0] == col_name for col in df.columns)
    else:
        return col_name in df.columns
        
def get_column(df, col_name):
    """데이터프레임에서 컬럼 데이터 가져오기(다중 인덱스 지원)"""
    if isinstance(df.columns, pd.MultiIndex):
        cols = [col for col in df.columns if col[0] == col_name]
        if cols:
            return df[cols[0]]
        return None
    else:
        if col_name in df.columns:
            return df[col_name]
        return None

# 페이지 설정
st.set_page_config(
    page_title="한국 주식 트레이딩 봇",
    page_icon="📈",
    layout="wide"
)

# 세션 상태 초기화
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = StockData()
    
if 'technical_indicators' not in st.session_state:
    st.session_state.technical_indicators = TechnicalIndicators()
    
if 'trading_signals' not in st.session_state:
    st.session_state.trading_signals = TradingSignals()
    
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# 헤더
st.title("한국 주식 트레이딩 봇")
st.markdown("실시간 주가 모니터링, 기술적 지표 계산, 자동 매매 신호 생성")

# 사이드바
with st.sidebar:
    st.header("설정")
    
    # 자동 업데이트 설정
    auto_update = st.checkbox("자동 업데이트", value=True)
    update_interval = st.slider("업데이트 주기(초)", min_value=5, max_value=60, value=10)
    
    # 종목 선택
    selected_codes = st.multiselect(
        "모니터링할 종목",
        options=STOCK_CODES,
        default=STOCK_CODES[:6]  # 기본적으로 처음 6개 종목 선택
    )
    
    
    # 수동 업데이트 버튼 (더 눈에 띄게 만들기)
    update_col1, update_col2 = st.columns([3, 1])
    with update_col1:
        if st.button("📊 데이터 업데이트", use_container_width=True):
            with st.spinner("데이터 업데이트 중..."):
                st.session_state.stock_data.update_data(force=True)
                st.session_state.last_update = datetime.now()
                if st.session_state.stock_data.sample_data_loaded:
                    st.warning("실제 데이터를 가져오지 못해 샘플 데이터를 사용합니다.")
                else:
                    st.success(f"데이터 업데이트 완료: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with update_col2:
        st.write(f"마지막 업데이트: {st.session_state.last_update.strftime('%H:%M:%S')}")

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["실시간 모니터링", "기술적 지표", "백테스팅", "모의 트레이딩"])

# 자동 업데이트
if auto_update:
    current_time = datetime.now()
    if (current_time - st.session_state.last_update).total_seconds() > update_interval:
        with st.spinner("데이터 업데이트 중..."):
            # 실시간 데이터 시뮬레이션
            st.session_state.stock_data.simulate_realtime_data()
            st.session_state.last_update = current_time

# 탭 1: 실시간 모니터링
with tab1:
    st.header("실시간 주가 모니터링")
    
    # 종목별 최신 데이터 표시
    latest_data = []
    
    for code in selected_codes:
        if code in st.session_state.stock_data.data and not st.session_state.stock_data.data[code].empty:
            latest = st.session_state.stock_data.get_latest_price(code)
            
            if latest is not None:
                # 기술적 지표 계산
                df = st.session_state.stock_data.data[code].copy()
                df = st.session_state.technical_indicators.calculate_all(df)
                df = st.session_state.trading_signals.generate_signals(df)
                
                # 최신 매매 신호
                signal_info = st.session_state.trading_signals.get_latest_signals(df)
                
                if signal_info:
                    # latest는 이미 get_latest_price 함수에서 딕셔너리로 변환됨
                    # 안전하게 값 가져오기
                    close_value = latest.get('Close', 0)
                    volume_value = latest.get('Volume', 0)
                    change_value = latest.get('Change', 0)
                    
                    latest_data.append({
                        '종목코드': code,
                        '현재가': f"{int(close_value):,}원",  # 원화 단위로 표시, 천 단위 구분자 추가
                        '변동률(%)': f"{change_value * 100:.2f}%",
                        '거래량': f"{int(volume_value):,}",  # 천 단위 구분자 추가
                        '매매신호': '매수' if signal_info['trading_decision'] == 1 else ('매도' if signal_info['trading_decision'] == -1 else '홀드'),
                        '신호강도': signal_info['signal_strength']
                    })
    
    if latest_data:
        df_latest = pd.DataFrame(latest_data)
        st.dataframe(df_latest)
    else:
        # 데이터가 없을 때 더 명확한 메시지와 해결 방법 제공
        st.error("데이터가 없습니다!")
        st.info("다음 방법을 시도해 보세요:")
        st.markdown("""
        1. 상단의 **📊 데이터 업데이트** 버튼을 클릭하여 데이터를 업데이트하세요.
        2. 인터넷 연결을 확인하세요.
        3. 다른 종목 코드를 선택해 보세요.
        """)
        
        # 샘플 데이터 사용 중인 경우 알림
        if hasattr(st.session_state.stock_data, 'sample_data_loaded') and st.session_state.stock_data.sample_data_loaded:
            st.warning("현재 실제 데이터를 가져오지 못해 샘플 데이터를 사용 중입니다.")
    
    # 주가 차트
    st.subheader("주가 차트")
    
    selected_code = st.selectbox("종목 선택", options=selected_codes)
    
    if selected_code in st.session_state.stock_data.data and not st.session_state.stock_data.data[selected_code].empty:
        df = st.session_state.stock_data.data[selected_code].copy()
        
        # 기간 선택
        days = st.slider("표시할 기간(일)", min_value=5, max_value=100, value=30)
        df_display = df.tail(days)
        
        # Plotly로 캔들스틱 차트 생성
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('주가', '거래량'),
                           row_heights=[0.7, 0.3])
        
        # 캔들스틱 차트
        # 가격 데이터를 정수로 변환하여 표시
        open_data = get_column(df_display, 'Open')
        high_data = get_column(df_display, 'High')
        low_data = get_column(df_display, 'Low')
        close_data = get_column(df_display, 'Close')
        volume_data = get_column(df_display, 'Volume')
        
        if open_data is not None and high_data is not None and low_data is not None and close_data is not None:
            open_int = open_data.round().astype(int)
            high_int = high_data.round().astype(int)
            low_int = low_data.round().astype(int)
            close_int = close_data.round().astype(int)
            
            fig.add_trace(
                go.Candlestick(
                    x=df_display.index,
                    open=open_int,
                    high=high_int,
                    low=low_int,
                    close=close_int,
                    name='주가'
                ),
                row=1, col=1
            )
            
            # 거래량 차트
            if volume_data is not None:
                fig.add_trace(
                    go.Bar(
                        x=df_display.index,
                        y=volume_data,
                        name='거래량'
                    ),
                    row=2, col=1
                )
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{selected_code} 주가 차트',
            xaxis_title='날짜',
            yaxis_title='가격',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"{selected_code} 데이터가 없습니다!")
        st.info("상단의 **📊 데이터 업데이트** 버튼을 클릭하여 데이터를 업데이트하세요.")

# 탭 2: 기술적 지표
with tab2:
    st.header("기술적 지표 분석")
    
    # 종목 선택
    selected_code_t2 = st.selectbox("종목 선택", options=selected_codes, key='tab2_code')
    
    if selected_code_t2 in st.session_state.stock_data.data and not st.session_state.stock_data.data[selected_code_t2].empty:
        # 데이터 준비
        df = st.session_state.stock_data.data[selected_code_t2].copy()
        df = st.session_state.technical_indicators.calculate_all(df)
        
        # 기간 선택
        days_t2 = st.slider("표시할 기간(일)", min_value=5, max_value=100, value=30, key='tab2_days')
        df_display = df.tail(days_t2)
        
        # 기술적 지표 선택
        indicator = st.selectbox(
            "기술적 지표 선택",
            options=["이동평균선(SMA)", "RSI", "MACD", "볼린저 밴드", "스토캐스틱"]
        )
        
        # SMA 지표가 존재하는지 확인
        has_sma5 = has_column(df_display, 'SMA_5')
        has_sma20 = has_column(df_display, 'SMA_20')
        
        if indicator == "이동평균선(SMA)" and has_sma5 and has_sma20:
            fig = go.Figure()
            
            # 캔들스틱 차트
            # 가격 데이터를 정수로 변환하여 표시
            open_data = get_column(df_display, 'Open')
            high_data = get_column(df_display, 'High')
            low_data = get_column(df_display, 'Low')
            close_data = get_column(df_display, 'Close')
            
            if open_data is not None and high_data is not None and low_data is not None and close_data is not None:
                open_int = open_data.round().astype(int)
                high_int = high_data.round().astype(int)
                low_int = low_data.round().astype(int)
                close_int = close_data.round().astype(int)
                
                fig.add_trace(
                    go.Candlestick(
                        x=df_display.index,
                        open=open_int,
                        high=high_int,
                        low=low_int,
                        close=close_int,
                        name='주가'
                    )
                )
                
                # 이동평균선
                sma5_data = get_column(df_display, 'SMA_5')
                sma20_data = get_column(df_display, 'SMA_20')
                
                if sma5_data is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=sma5_data,
                            name='SMA 5',
                            line=dict(color='blue', width=1)
                        )
                    )
                
                if sma20_data is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=sma20_data,
                            name='SMA 20',
                            line=dict(color='red', width=1)
                        )
                    )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{selected_code_t2} 이동평균선(SMA)',
                xaxis_title='날짜',
                yaxis_title='가격',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 설명
            st.markdown("""
            **이동평균선(SMA) 분석**
            
            이동평균선은 일정 기간 동안의 주가 평균을 나타내는 지표입니다. 단기 이동평균선(SMA 5)이 장기 이동평균선(SMA 20)을 상향 돌파할 때 매수 신호(골든 크로스)가 발생하고, 하향 돌파할 때 매도 신호(데드 크로스)가 발생합니다.
            """)
        
        elif indicator == "RSI" and has_column(df_display, 'RSI'):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.1,
                               row_heights=[0.7, 0.3])
            
            # 캔들스틱 차트
            # 가격 데이터를 정수로 변환하여 표시
            open_data = get_column(df_display, 'Open')
            high_data = get_column(df_display, 'High')
            low_data = get_column(df_display, 'Low')
            close_data = get_column(df_display, 'Close')
            
            if open_data is not None and high_data is not None and low_data is not None and close_data is not None:
                open_int = open_data.round().astype(int)
                high_int = high_data.round().astype(int)
                low_int = low_data.round().astype(int)
                close_int = close_data.round().astype(int)
                
                fig.add_trace(
                    go.Candlestick(
                        x=df_display.index,
                        open=open_int,
                        high=high_int,
                        low=low_int,
                        close=close_int,
                        name='주가'
                    ),
                    row=1, col=1
                )
                
                # RSI 차트
                rsi_data = get_column(df_display, 'RSI')
                
                if rsi_data is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=rsi_data,
                            name='RSI',
                            line=dict(color='purple', width=1)
                        ),
                        row=2, col=1
                    )
            
            # 과매수/과매도 라인
            fig.add_trace(
                go.Scatter(
                    x=[df_display.index[0], df_display.index[-1]],
                    y=[70, 70],
                    name='과매수(70)',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df_display.index[0], df_display.index[-1]],
                    y=[30, 30],
                    name='과매도(30)',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{selected_code_t2} RSI(상대강도지수)',
                xaxis_title='날짜',
                yaxis_title='가격',
                xaxis2_title='날짜',
                yaxis2_title='RSI',
                height=600
            )
            
            # RSI y축 범위 설정
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 설명
            st.markdown("""
            **RSI(상대강도지수) 분석**
            
            RSI는 주가의 상승 압력과 하락 압력 간의 상대적인 강도를 나타내는 지표입니다. 일반적으로 RSI가 70 이상이면 과매수 상태로 매도 신호로 해석하고, 30 이하면 과매도 상태로 매수 신호로 해석합니다.
            """)
        
        elif indicator == "스토캐스틱" and has_column(df_display, 'Stochastic_K') and has_column(df_display, 'Stochastic_D'):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.1,
                               row_heights=[0.7, 0.3])
            
            # 캔들스틱 차트
            # 가격 데이터를 정수로 변환하여 표시
            open_data = get_column(df_display, 'Open')
            high_data = get_column(df_display, 'High')
            low_data = get_column(df_display, 'Low')
            close_data = get_column(df_display, 'Close')
            
            if open_data is not None and high_data is not None and low_data is not None and close_data is not None:
                open_int = open_data.round().astype(int)
                high_int = high_data.round().astype(int)
                low_int = low_data.round().astype(int)
                close_int = close_data.round().astype(int)
                
                fig.add_trace(
                    go.Candlestick(
                        x=df_display.index,
                        open=open_int,
                        high=high_int,
                        low=low_int,
                        close=close_int,
                        name='주가'
                    ),
                    row=1, col=1
                )
                
                # 스토캐스틱 차트
                stochastic_k = get_column(df_display, 'Stochastic_K')
                stochastic_d = get_column(df_display, 'Stochastic_D')
                
                if stochastic_k is not None and stochastic_d is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=stochastic_k,
                            name='%K',
                            line=dict(color='blue', width=1)
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=stochastic_d,
                            name='%D',
                            line=dict(color='red', width=1)
                        ),
                        row=2, col=1
                    )
            
            # 과매수/과매도 라인
            overbought = 80  # 설정 파일에서 가져오는 것이 좋음
            oversold = 20    # 설정 파일에서 가져오는 것이 좋음
            
            fig.add_trace(
                go.Scatter(
                    x=[df_display.index[0], df_display.index[-1]],
                    y=[overbought, overbought],
                    name=f'과매수({overbought})',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df_display.index[0], df_display.index[-1]],
                    y=[oversold, oversold],
                    name=f'과매도({oversold})',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'{selected_code_t2} 스토캐스틱 오실레이터',
                xaxis_title='날짜',
                yaxis_title='가격',
                xaxis2_title='날짜',
                yaxis2_title='스토캐스틱',
                height=600
            )
            
            # 스토캐스틱 y축 범위 설정
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 설명
            st.markdown("""
            **스토캐스틱 오실레이터 분석**
            
            스토캐스틱 오실레이터는 주가의 모멘텀을 측정하는 지표로, %K와 %D 두 개의 선으로 구성됩니다.
            일반적으로 %K가 %D를 상향 돌파할 때 매수 신호로, 하향 돌파할 때 매도 신호로 해석합니다.
            또한 지표가 80 이상이면 과매수 상태, 20 이하면 과매도 상태로 간주합니다.
            """)
        else:
            st.warning("선택한 기술적 지표 데이터가 없습니다.")
            st.info("상단의 **📊 데이터 업데이트** 버튼을 클릭하여 데이터를 업데이트하세요.")

# 탭 3: 백테스팅
with tab3:
    st.header("백테스팅")
    
    # 백테스팅 설정
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_code = st.selectbox("백테스팅할 종목", options=STOCK_CODES)
        initial_capital = st.number_input("초기 자본금", min_value=1000000, max_value=100000000, value=10000000, step=1000000)
    
    with col2:
        start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("종료일", value=datetime.now())
    
    # 백테스팅 실행 버튼
    if st.button("백테스팅 실행"):
        if backtest_code in st.session_state.stock_data.data:
            with st.spinner("백테스팅 실행 중..."):
                # 데이터 준비
                df = st.session_state.stock_data.data[backtest_code].copy()
                
                # 결측치 확인
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning(f"데이터에 결측치가 있습니다: {missing_values}")
                
                # 기간 필터링
                mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
                df = df.loc[mask]
                
                if not df.empty:
                    # 결측치 처리
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in df.columns:  # 먼저 컬럼 존재 여부를 확인
                            # if df[col].isnull().any():  # 결측치가 있는지 확인
                            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')  # 결측치 처리
                            st.info(f"{col} 컬럼의 결측치를 처리했습니다.")
                                                    
                    # 기술적 지표 계산
                    df = st.session_state.technical_indicators.calculate_all(df)
                    df = st.session_state.trading_signals.generate_signals(df)
                    
                    # 백테스팅 수행 전 데이터 확인
                    # 다중 인덱스 처리
                    close_null = False
                    trading_decision_null = False
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        # MultiIndex인 경우
                        close_cols = [col for col in df.columns if col[0] == 'Close']
                        trading_cols = [col for col in df.columns if col[0] == 'Trading_Decision']
                    
                    if close_null or trading_decision_null:
                        st.warning("백테스팅에 필요한 데이터에 결측치가 있습니다. 결과가 정확하지 않을 수 있습니다.")
                    
                    # 백테스팅 수행
                    backtest_result = st.session_state.trading_signals.backtest(df, initial_capital)
                    
                    if backtest_result:
                        # 결과 표시
                        st.subheader("백테스팅 결과 요약")
                        
                        # NaN 값 확인 및 처리
                        check_keys = [
                            'final_capital', 'total_return_pct', 'annual_return_pct',
                            'volatility_pct', 'sharpe_ratio', 'max_drawdown_pct',
                            'win_rate_pct', 'avg_profit_pct', 'avg_loss_pct', 'profit_loss_ratio'
                        ]
                        for key in check_keys:
                            if key in backtest_result and (np.isnan(backtest_result[key]) or np.isinf(backtest_result[key])):
                                st.warning(f"백테스팅 결과의 {key} 값이 NaN 또는 Inf입니다. 0으로 대체합니다.")
                                backtest_result[key] = 0
                        
                        summary_data = {
                            '항목': [
                                '초기 자본금',
                                '최종 자본금',
                                '총 수익률',
                                '연간 수익률',
                                '변동성',
                                '샤프 비율',
                                '거래 횟수',
                                '최대 낙폭',
                                '승률',
                                '평균 수익',
                                '평균 손실',
                                '손익비'
                            ],
                            '값': [
                                f"{int(backtest_result['initial_capital']):,}원",
                                f"{int(backtest_result['final_capital']):,}원",
                                f"{backtest_result['total_return_pct']:.2f}%",
                                f"{backtest_result['annual_return_pct']:.2f}%",
                                f"{backtest_result['volatility_pct']:.2f}%",
                                f"{backtest_result['sharpe_ratio']:.2f}",
                                f"{backtest_result['num_trades']}회",
                                f"{backtest_result['max_drawdown_pct']:.2f}%",
                                f"{backtest_result['win_rate_pct']:.2f}%",
                                f"{backtest_result['avg_profit_pct']:.2f}%",
                                f"{backtest_result['avg_loss_pct']:.2f}%",
                                f"{backtest_result['profit_loss_ratio']:.2f}"
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.table(summary_df)
                        
                        # 그래프 표시
                        st.subheader("수익률 그래프")
                        
                        backtest_data = backtest_result['backtest_data']
                        
                        fig = go.Figure()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=backtest_data.index,
                                # y=backtest_data['Cumulative_Returns'],
                                y=backtest_data[('Cumulative_Returns')],
                                name='Buy & Hold'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=backtest_data.index,
                                # y=backtest_data['Strategy_Cumulative_Returns'],
                                y=backtest_data[('Strategy_Cumulative_Returns')],
                                name='Strategy'
                            )
                        )
                        
                        fig.update_layout(
                            title=f'{backtest_code} 백테스팅 결과 - 누적 수익률',
                            xaxis_title='날짜',
                            yaxis_title='누적 수익률',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("백테스팅 결과를 계산할 수 없습니다.")
                else:
                    st.error("선택한 기간에 데이터가 없습니다.")
        else:
            st.error(f"{backtest_code} 데이터가 없습니다!")
            st.info("상단의 **📊 데이터 업데이트** 버튼을 클릭하여 데이터를 업데이트하세요.")
            
            # 샘플 데이터 생성 버튼 추가
            if st.button("샘플 데이터로 테스트"):
                with st.spinner("샘플 데이터 생성 중..."):
                    # 샘플 데이터 생성
                    sample_df = st.session_state.stock_data._generate_sample_data(backtest_code)
                    st.session_state.stock_data.data[backtest_code] = sample_df
                    st.session_state.stock_data.last_update[backtest_code] = datetime.now()
                    st.session_state.stock_data.sample_data_loaded = True
                    st.success(f"{backtest_code} 샘플 데이터 생성 완료! 이제 백테스팅을 실행해 보세요.")

# 탭 4: 모의 트레이딩
with tab4:
    st.header("모의 계좌 트레이딩")
    
    # KIS API 인스턴스 생성
    if 'kis_api' not in st.session_state:
        from kis_api import KoreaInvestmentAPI
        st.session_state.kis_api = KoreaInvestmentAPI()
    
    # 자동 매매 설정 초기화
    if 'auto_trading_enabled' not in st.session_state:
        st.session_state.auto_trading_enabled = False
    
    if 'auto_trading_settings' not in st.session_state:
        st.session_state.auto_trading_settings = {
            'investment_amount_per_trade': 1000000,  # 1회 투자금액 (100만원)
            'max_positions': 5,  # 최대 포지션 수
            'stop_loss_pct': 3.0,  # 손절매 비율 (%)
            'take_profit_pct': 5.0,  # 익절매 비율 (%)
            'last_check_time': None,  # 마지막 신호 체크 시간
            'check_interval': 60  # 신호 체크 주기 (초)
        }
    
    # 탭 생성
    trade_tab1, trade_tab2, trade_tab3 = st.tabs(["계좌 정보", "수동 매매", "자동 매매"])
    
    # 계좌 정보 조회
    if 'account_balance' not in st.session_state or st.button("계좌 정보 새로고침", key="refresh_account"):
        with st.spinner("계좌 정보 조회 중..."):
            st.session_state.account_balance = st.session_state.kis_api.get_account_balance()
    
    # 탭 1: 계좌 정보
    with trade_tab1:
        # 계좌 정보 표시
        if st.session_state.account_balance:
            # 계좌 요약 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 평가금액", f"{int(st.session_state.account_balance['total_balance']):,}원")
            with col2:
                st.metric("예수금", f"{int(st.session_state.account_balance['cash_balance']):,}원")
            with col3:
                profit_loss = st.session_state.account_balance['profit_loss']
                profit_loss_rate = st.session_state.account_balance['profit_loss_rate']
                delta_color = "normal" if profit_loss == 0 else ("inverse" if profit_loss < 0 else "normal")
                st.metric("평가손익", f"{int(profit_loss):,}원", f"{profit_loss_rate:.2f}%", delta_color=delta_color)
            
            # 보유 종목 정보
            st.subheader("보유 종목")
            if st.session_state.account_balance['holdings']:
                holdings_data = []
                for holding in st.session_state.account_balance['holdings']:
                    holdings_data.append({
                        '종목코드': holding['code'],
                        '종목명': holding['name'],
                        '보유수량': f"{holding['quantity']:,}주",
                        '매입가': f"{int(holding['avg_price']):,}원",
                        '현재가': f"{int(holding['current_price']):,}원",
                        '평가금액': f"{int(holding['value']):,}원",
                        '평가손익': f"{int(holding['profit_loss']):,}원",
                        '수익률': f"{holding['profit_loss_rate']:.2f}%"
                    })
                
                df_holdings = pd.DataFrame(holdings_data)
                st.dataframe(df_holdings)
            else:
                st.info("보유 종목이 없습니다.")
        else:
            st.error("계좌 정보를 불러올 수 없습니다.")
        
        # 주문 내역 조회
        st.subheader("주문 내역")
        
        if st.button("주문 내역 조회", key="get_order_history"):
            with st.spinner("주문 내역 조회 중..."):
                order_history = st.session_state.kis_api.get_order_history()
                
                if order_history:
                    order_data = []
                    for order in order_history:
                        order_data.append({
                            '주문일자': order['order_date'],
                            '주문시각': order['order_time'],
                            '종목코드': order['code'],
                            '종목명': order['name'],
                            '주문구분': order['order_type'],
                            '주문수량': f"{order['quantity']:,}주",
                            '주문가격': f"{int(order['price']):,}원",
                            '체결수량': f"{order['executed_quantity']:,}주",
                            '체결가격': f"{int(order['executed_price']):,}원" if order['executed_price'] > 0 else "-",
                            '주문상태': order['order_status'],
                            '주문번호': order['order_no']
                        })
                    
                    df_orders = pd.DataFrame(order_data)
                    st.dataframe(df_orders)
                else:
                    st.info("주문 내역이 없습니다.")
    
    # 탭 2: 수동 매매
    with trade_tab2:
        st.subheader("수동 주문 실행")
        
        # 종목 선택
        order_code = st.selectbox("종목 선택", options=STOCK_CODES, key="order_code")
        
        # 현재가 조회
        current_price = None
        if order_code:
            with st.spinner("현재가 조회 중..."):
                price_info = st.session_state.kis_api.get_stock_current_price(order_code)
                if price_info:
                    current_price = price_info['Close']
                    st.info(f"현재가: {int(current_price):,}원")
        
        # 주문 양식
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 매수 주문")
            buy_quantity = st.number_input("매수 수량", min_value=1, value=1, step=1, key="buy_quantity")
            buy_price = st.number_input("매수 가격", min_value=0, value=int(current_price) if current_price else 0, step=10, key="buy_price")
            buy_condition = st.radio("주문 조건", ["지정가", "시장가"], key="buy_condition")
            
            if st.button("매수 주문", key="buy_order"):
                with st.spinner("매수 주문 실행 중..."):
                    order_condition = "00" if buy_condition == "지정가" else "01"
                    price = buy_price if buy_condition == "지정가" else 0
                    
                    result = st.session_state.kis_api.place_order(
                        code=order_code,
                        order_type="1",  # 매수
                        quantity=buy_quantity,
                        price=price,
                        order_condition=order_condition
                    )
                    
                    if result:
                        st.success(f"매수 주문이 실행되었습니다. 주문번호: {result['order_no']}")
                        # 계좌 정보 새로고침
                        st.session_state.account_balance = st.session_state.kis_api.get_account_balance()
                    else:
                        st.error("매수 주문 실행에 실패했습니다.")
        
        with col2:
            st.markdown("### 매도 주문")
            sell_quantity = st.number_input("매도 수량", min_value=1, value=1, step=1, key="sell_quantity")
            sell_price = st.number_input("매도 가격", min_value=0, value=int(current_price) if current_price else 0, step=10, key="sell_price")
            sell_condition = st.radio("주문 조건", ["지정가", "시장가"], key="sell_condition")
            
            if st.button("매도 주문", key="sell_order"):
                with st.spinner("매도 주문 실행 중..."):
                    order_condition = "00" if sell_condition == "지정가" else "01"
                    price = sell_price if sell_condition == "지정가" else 0
                    
                    result = st.session_state.kis_api.place_order(
                        code=order_code,
                        order_type="2",  # 매도
                        quantity=sell_quantity,
                        price=price,
                        order_condition=order_condition
                    )
                    
                    if result:
                        st.success(f"매도 주문이 실행되었습니다. 주문번호: {result['order_no']}")
                        # 계좌 정보 새로고침
                        st.session_state.account_balance = st.session_state.kis_api.get_account_balance()
                    else:
                        st.error("매도 주문 실행에 실패했습니다.")
    
    # 탭 3: 자동 매매
    with trade_tab3:
        st.subheader("자동 매매 설정")
        
        # 자동 매매 활성화/비활성화
        auto_trading_enabled = st.toggle("자동 매매 활성화", value=st.session_state.auto_trading_enabled)
        
        if auto_trading_enabled != st.session_state.auto_trading_enabled:
            st.session_state.auto_trading_enabled = auto_trading_enabled
            if auto_trading_enabled:
                st.success("자동 매매가 활성화되었습니다.")
                # 마지막 체크 시간 초기화
                st.session_state.auto_trading_settings['last_check_time'] = datetime.now()
            else:
                st.warning("자동 매매가 비활성화되었습니다.")
        
        # 자동 매매 설정
        st.markdown("### 자동 매매 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "1회 투자금액 (원)",
                min_value=100000,
                max_value=10000000,
                value=st.session_state.auto_trading_settings['investment_amount_per_trade'],
                step=100000,
                key="investment_amount"
            )
            
            max_positions = st.number_input(
                "최대 포지션 수",
                min_value=1,
                max_value=10,
                value=st.session_state.auto_trading_settings['max_positions'],
                step=1,
                key="max_positions"
            )
        
        with col2:
            stop_loss_pct = st.number_input(
                "손절매 비율 (%)",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.auto_trading_settings['stop_loss_pct'],
                step=0.5,
                key="stop_loss_pct"
            )
            
            take_profit_pct = st.number_input(
                "익절매 비율 (%)",
                min_value=1.0,
                max_value=20.0,
                value=st.session_state.auto_trading_settings['take_profit_pct'],
                step=0.5,
                key="take_profit_pct"
            )
        
        check_interval = st.slider(
            "신호 체크 주기 (초)",
            min_value=30,
            max_value=300,
            value=st.session_state.auto_trading_settings['check_interval'],
            step=30,
            key="check_interval"
        )
        
        # 설정 저장
        if st.button("설정 저장", key="save_settings"):
            st.session_state.auto_trading_settings.update({
                'investment_amount_per_trade': investment_amount,
                'max_positions': max_positions,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'check_interval': check_interval
            })
            st.success("자동 매매 설정이 저장되었습니다.")
        
        # 매매 전략 선택
        st.markdown("### 매매 전략 선택")
        
        trading_strategies = {
            "SMA 크로스오버": "sma_crossover",
            "RSI 과매수/과매도": "rsi_bounds",
            "MACD 크로스오버": "macd_crossover",
            "볼린저 밴드 돌파": "bollinger_breakout",
            "스토캐스틱 크로스오버": "stochastic_crossover",
            "모든 전략 (AND)": "all_strategies_and",
            "모든 전략 (OR)": "all_strategies_or"
        }
        
        selected_strategies = st.multiselect(
            "사용할 매매 전략",
            options=list(trading_strategies.keys()),
            default=["SMA 크로스오버", "RSI 과매수/과매도"],
            key="selected_strategies"
        )
        
        # 자동 매매 상태 표시
        st.markdown("### 자동 매매 상태")
        
        # 자동 매매 로직 실행
        if st.session_state.auto_trading_enabled:
            current_time = datetime.now()
            last_check_time = st.session_state.auto_trading_settings.get('last_check_time')
            
            # 체크 주기마다 신호 확인
            if last_check_time is None or (current_time - last_check_time).total_seconds() >= st.session_state.auto_trading_settings['check_interval']:
                with st.spinner("매매 신호 확인 중..."):
                    # 데이터 업데이트
                    st.session_state.stock_data.update_data()
                    
                    # 선택된 전략에 따른 매매 신호 생성
                    signals = {}
                    for code in STOCK_CODES:
                        if code in st.session_state.stock_data.data and not st.session_state.stock_data.data[code].empty:
                            # 기술적 지표 계산
                            df = st.session_state.technical_indicators.calculate_all(st.session_state.stock_data.data[code])
                            
                            # 매매 신호 생성
                            df_with_signals = st.session_state.trading_signals.generate_signals(df)
                            
                            # 최신 신호 저장
                            signals[code] = st.session_state.trading_signals.get_latest_signals(df_with_signals)
                    
                    # 계좌 정보 조회
                    account_balance = st.session_state.kis_api.get_account_balance()
                    st.session_state.account_balance = account_balance
                    
                    if account_balance:
                        # 현재 보유 종목 확인
                        holdings = {holding['code']: holding for holding in account_balance['holdings']}
                        
                        # 매매 신호에 따라 주문 실행
                        for code, signal in signals.items():
                            if signal is None:
                                continue
                            
                            # 매매 결정
                            decision = signal['trading_decision']
                            
                            # 매수 신호
                            if decision == 1:
                                # 이미 보유 중인지 확인
                                if code in holdings:
                                    st.info(f"{code} 종목은 이미 보유 중입니다.")
                                    continue
                                
                                # 최대 포지션 수 확인
                                if len(holdings) >= st.session_state.auto_trading_settings['max_positions']:
                                    st.warning(f"최대 포지션 수({st.session_state.auto_trading_settings['max_positions']})에 도달했습니다.")
                                    continue
                                
                                # 투자 금액 확인
                                if account_balance['cash_balance'] < st.session_state.auto_trading_settings['investment_amount_per_trade']:
                                    st.warning("투자 가능한 예수금이 부족합니다.")
                                    continue
                                
                                # 현재가 조회
                                price_info = st.session_state.kis_api.get_stock_current_price(code)
                                if not price_info:
                                    st.error(f"{code} 종목의 현재가를 조회할 수 없습니다.")
                                    continue
                                
                                current_price = price_info['Close']
                                
                                # 매수 수량 계산
                                quantity = int(st.session_state.auto_trading_settings['investment_amount_per_trade'] / current_price)
                                
                                if quantity > 0:
                                    # 매수 주문 실행
                                    result = st.session_state.kis_api.place_order(
                                        code=code,
                                        order_type="1",  # 매수
                                        quantity=quantity,
                                        price=0,  # 시장가 주문
                                        order_condition="01"  # 시장가
                                    )
                                    
                                    if result:
                                        st.success(f"{code} 종목 매수 주문이 실행되었습니다. 주문번호: {result['order_no']}")
                                    else:
                                        st.error(f"{code} 종목 매수 주문 실행에 실패했습니다.")
                            
                            # 매도 신호
                            elif decision == -1:
                                # 보유 중인지 확인
                                if code not in holdings:
                                    continue
                                
                                # 보유 수량 확인
                                quantity = holdings[code]['quantity']
                                
                                if quantity > 0:
                                    # 매도 주문 실행
                                    result = st.session_state.kis_api.place_order(
                                        code=code,
                                        order_type="2",  # 매도
                                        quantity=quantity,
                                        price=0,  # 시장가 주문
                                        order_condition="01"  # 시장가
                                    )
                                    
                                    if result:
                                        st.success(f"{code} 종목 매도 주문이 실행되었습니다. 주문번호: {result['order_no']}")
                                    else:
                                        st.error(f"{code} 종목 매도 주문 실행에 실패했습니다.")
                    
                    # 마지막 체크 시간 업데이트
                    st.session_state.auto_trading_settings['last_check_time'] = current_time
            
            # 다음 체크 시간 표시
            if last_check_time:
                next_check_time = last_check_time + timedelta(seconds=st.session_state.auto_trading_settings['check_interval'])
                time_remaining = max(0, (next_check_time - current_time).total_seconds())
                
                st.info(f"마지막 신호 체크: {last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.info(f"다음 신호 체크: {next_check_time.strftime('%Y-%m-%d %H:%M:%S')} (약 {int(time_remaining)}초 후)")
                
                # 현재 신호 상태 표시
                st.subheader("현재 매매 신호")
                
                signal_data = []
                for code in STOCK_CODES:
                    if code in st.session_state.stock_data.data and not st.session_state.stock_data.data[code].empty:
                        # 기술적 지표 계산
                        df = st.session_state.technical_indicators.calculate_all(st.session_state.stock_data.data[code])
                        
                        # 매매 신호 생성
                        df_with_signals = st.session_state.trading_signals.generate_signals(df)
                        
                        # 최신 신호 저장
                        signal = st.session_state.trading_signals.get_latest_signals(df_with_signals)
                        
                        if signal:
                            signal_data.append({
                                '종목코드': code,
                                '매매신호': '매수' if signal['trading_decision'] == 1 else ('매도' if signal['trading_decision'] == -1 else '홀드'),
                                '신호강도': signal['signal_strength'],
                                '현재가': f"{int(signal['price']):,}원"
                            })
                
                if signal_data:
                    df_signals = pd.DataFrame(signal_data)
                    st.dataframe(df_signals)
                else:
                    st.info("현재 매매 신호가 없습니다.")
        else:
            st.warning("자동 매매가 비활성화되어 있습니다. 활성화하려면 상단의 토글 버튼을 클릭하세요.")

# 앱 실행 방법 안내
st.sidebar.markdown("---")
st.sidebar.subheader("앱 실행 방법")
st.sidebar.code("streamlit run app.py")

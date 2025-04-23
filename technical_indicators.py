import pandas as pd
import numpy as np
import logging
from config import TECHNICAL_INDICATORS
from utils import handle_missing_values

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('technical_indicators')

class TechnicalIndicators:
    def __init__(self):
        self.config = TECHNICAL_INDICATORS
        logger.info("기술적 지표 계산기 초기화 완료")
    
    def calculate_all(self, df):
        """모든 기술적 지표 계산"""
        if df is None or df.empty:
            logger.warning("데이터가 비어있어 기술적 지표를 계산할 수 없습니다.")
            return df
        
        # 데이터 복사
        result = df.copy()
        
        # 결측치 확인
        missing_values = result.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"기술적 지표 계산 전 결측치 발견: {missing_values}")
            
            # 중앙 집중식 결측치 처리 함수 사용
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            result = handle_missing_values(result, method='ffill_bfill', columns=required_columns)
        
        # 각 지표 계산
        result = self.calculate_sma(result)
        result = self.calculate_rsi(result)
        result = self.calculate_macd(result)
        result = self.calculate_bollinger_bands(result)
        result = self.calculate_stochastic(result)
        
        # 계산 후 결측치 확인
        post_missing = result.isnull().sum()
        if post_missing.sum() > 0:
            logger.warning(f"기술적 지표 계산 후 결측치 발견: {post_missing}")
        
        return result
    
    def calculate_sma(self, df):
        """단순 이동평균선(Simple Moving Average) 계산"""
        try:
            short_period = self.config['SMA']['short_period']
            long_period = self.config['SMA']['long_period']
            
            df[f'SMA_{short_period}'] = df['Close'].rolling(window=short_period).mean()
            df[f'SMA_{long_period}'] = df['Close'].rolling(window=long_period).mean()
            
            # 골든 크로스 / 데드 크로스 신호
            df['SMA_Signal'] = 0
            
            # NaN 값 처리
            valid_rows = df[[f'SMA_{short_period}', f'SMA_{long_period}']].notna().all(axis=1)
            
            # 유효한 행에 대해서만 신호 생성
            df.loc[valid_rows & (df[f'SMA_{short_period}'] > df[f'SMA_{long_period}']), 'SMA_Signal'] = 1  # 골든 크로스 (매수)
            df.loc[valid_rows & (df[f'SMA_{short_period}'] < df[f'SMA_{long_period}']), 'SMA_Signal'] = -1  # 데드 크로스 (매도)
            
            logger.info(f"SMA 계산 완료 (단기: {short_period}, 장기: {long_period})")
            
        except Exception as e:
            logger.error(f"SMA 계산 중 오류 발생: {str(e)}")
        
        return df
    
    def calculate_rsi(self, df):
        """상대강도지수(Relative Strength Index) 계산"""
        try:
            period = self.config['RSI']['period']
            overbought = self.config['RSI']['overbought']
            oversold = self.config['RSI']['oversold']
            
            # 가격 변화 계산
            delta = df['Close'].diff()
            
            # 상승/하락 구분
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 평균 상승/하락 계산
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # RS 계산 (상대강도)
            rs = avg_gain / avg_loss
            
            # RSI 계산
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # RSI 신호
            df['RSI_Signal'] = 0
            
            # NaN 값 처리
            valid_rows = df['RSI'].notna()
            
            # 유효한 행에 대해서만 신호 생성
            df.loc[valid_rows & (df['RSI'] < oversold), 'RSI_Signal'] = 1  # 과매도 (매수)
            df.loc[valid_rows & (df['RSI'] > overbought), 'RSI_Signal'] = -1  # 과매수 (매도)
            
            logger.info(f"RSI 계산 완료 (기간: {period}, 과매수: {overbought}, 과매도: {oversold})")
            
        except Exception as e:
            logger.error(f"RSI 계산 중 오류 발생: {str(e)}")
        
        return df
    
    def calculate_macd(self, df):
        """MACD(Moving Average Convergence Divergence) 계산"""
        try:
            fast_period = self.config['MACD']['fast_period']
            slow_period = self.config['MACD']['slow_period']
            signal_period = self.config['MACD']['signal_period']
            
            # EMA 계산
            ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
            
            # MACD 라인
            df['MACD'] = ema_fast - ema_slow
            
            # 시그널 라인
            df['MACD_Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
            
            # 히스토그램
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal_Line']
            
            # MACD 신호
            df['MACD_Signal'] = 0
            
            # NaN 값 처리
            valid_rows = df[['MACD', 'MACD_Signal_Line']].notna().all(axis=1)
            
            # 유효한 행에 대해서만 신호 생성
            df.loc[valid_rows & (df['MACD'] > df['MACD_Signal_Line']), 'MACD_Signal'] = 1  # 매수
            df.loc[valid_rows & (df['MACD'] < df['MACD_Signal_Line']), 'MACD_Signal'] = -1  # 매도
            
            logger.info(f"MACD 계산 완료 (빠른 기간: {fast_period}, 느린 기간: {slow_period}, 시그널 기간: {signal_period})")
            
        except Exception as e:
            logger.error(f"MACD 계산 중 오류 발생: {str(e)}")
        
        return df
    
    def calculate_bollinger_bands(self, df):
        """볼린저 밴드 계산"""
        try:
            period = self.config['Bollinger']['period']
            std_dev = self.config['Bollinger']['std_dev']
            
            # 중간 밴드 (SMA)
            df['BB_Middle'] = df['Close'].rolling(window=period).mean()
            
            # 표준편차
            df['BB_Std'] = df['Close'].rolling(window=period).std()
            
            # 상단 및 하단 밴드
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)
            
            # 볼린저 밴드 신호
            df['BB_Signal'] = 0
            
            # NaN 값 처리
            valid_rows = df[['Close', 'BB_Upper', 'BB_Lower']].notna().all(axis=1)
            
            # 유효한 행에 대해서만 신호 생성
            df.loc[valid_rows & (df['Close'] > df['BB_Upper']), 'BB_Signal'] = -1  # 상단 돌파 (매도)
            df.loc[valid_rows & (df['Close'] < df['BB_Lower']), 'BB_Signal'] = 1  # 하단 돌파 (매수)
            
            logger.info(f"볼린저 밴드 계산 완료 (기간: {period}, 표준편차: {std_dev})")
            
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류 발생: {str(e)}")
        
        return df
    
    def calculate_stochastic(self, df):
        """스토캐스틱 오실레이터(Stochastic Oscillator) 계산"""
        try:
            k_period = self.config['Stochastic']['k_period']
            k_slowing_period = self.config['Stochastic']['k_slowing_period']
            d_period = self.config['Stochastic']['d_period']
            overbought = self.config['Stochastic']['overbought']
            oversold = self.config['Stochastic']['oversold']
            
            # 필요한 데이터 확인
            if not all(col in df.columns for col in ['High', 'Low', 'Close']):
                logger.warning("스토캐스틱 계산에 필요한 컬럼(High, Low, Close)이 없습니다.")
                return df
            
            # 최근 k_period 동안의 최고가와 최저가 계산
            highest_high = df['High'].rolling(window=k_period).max()
            lowest_low = df['Low'].rolling(window=k_period).min()
            
            # %K 계산 (Fast Stochastic)
            # %K = (현재가 - 최저가) / (최고가 - 최저가) * 100
            df['Stochastic_K_Fast'] = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
            
            # %K 슬로잉 (Slow Stochastic)
            # %K = %K Fast의 k_slowing_period 이동평균
            df['Stochastic_K'] = df['Stochastic_K_Fast'].rolling(window=k_slowing_period).mean()
            
            # %D 계산 (Slow Stochastic의 이동평균)
            # %D = %K의 d_period 이동평균
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=d_period).mean()
            
            # 스토캐스틱 신호
            df['Stochastic_Signal'] = 0
            
            # NaN 값 처리
            valid_rows = df[['Stochastic_K', 'Stochastic_D']].notna().all(axis=1)
            
            # 과매수/과매도 신호 생성
            # 1. %K가 과매도 수준 아래로 내려갔다가 다시 올라올 때 매수 신호
            # 2. %K가 과매수 수준 위로 올라갔다가 다시 내려올 때 매도 신호
            # 3. %K와 %D의 교차도 신호로 사용 가능
            
            # 과매도 구간에서 %K가 %D를 상향 돌파할 때 매수 신호
            buy_signal = (df['Stochastic_K'].shift(1) < df['Stochastic_D'].shift(1)) & \
                         (df['Stochastic_K'] > df['Stochastic_D']) & \
                         (df['Stochastic_K'] < oversold + 10)  # 과매도 구간 근처
            
            # 과매수 구간에서 %K가 %D를 하향 돌파할 때 매도 신호
            sell_signal = (df['Stochastic_K'].shift(1) > df['Stochastic_D'].shift(1)) & \
                          (df['Stochastic_K'] < df['Stochastic_D']) & \
                          (df['Stochastic_K'] > overbought - 10)  # 과매수 구간 근처
            
            # 유효한 행에 대해서만 신호 생성
            df.loc[valid_rows & buy_signal, 'Stochastic_Signal'] = 1  # 매수
            df.loc[valid_rows & sell_signal, 'Stochastic_Signal'] = -1  # 매도
            
            logger.info(f"스토캐스틱 계산 완료 (K 기간: {k_period}, K 슬로잉: {k_slowing_period}, D 기간: {d_period})")
            
        except Exception as e:
            logger.error(f"스토캐스틱 계산 중 오류 발생: {str(e)}")
        
        return df

# 테스트 코드
if __name__ == "__main__":
    import yfinance as yf
    from pykrx import stock
    from datetime import datetime, timedelta
    
    # 테스트 데이터 가져오기
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    # yfinance로 데이터 가져오기
    code = '005930'
    yf_code = f"{code}.KS"
    df = yf.download(
        yf_code,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )
    
    # yfinance로 데이터를 가져오지 못한 경우 pykrx 사용
    if df.empty:
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        df = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, code)
        
        # 컬럼명 변경
        if not df.empty:
            df = df.rename(columns={
                '시가': 'Open',
                '고가': 'High',
                '저가': 'Low',
                '종가': 'Close',
                '거래량': 'Volume'
            })
            
            # Change 컬럼 추가
            df['Change'] = df['Close'].pct_change()
    
    # 기술적 지표 계산
    ti = TechnicalIndicators()
    result = ti.calculate_all(df)
    
    # 결과 출력
    print(result.tail())
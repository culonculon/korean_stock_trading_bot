import pandas as pd
import numpy as np
from pykrx import stock
import time
import os
from datetime import datetime, timedelta
import logging
from config import STOCK_CODES, UPDATE_INTERVAL
from utils import handle_missing_values
from database import StockDatabase
from kis_api import KoreaInvestmentAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_data')
class StockData:
    def __init__(self, db_path='data/stock_data.db'):
        self.data = {}
        self.last_update = {}
        self.sample_data_loaded = False
        
        # 데이터베이스 디렉토리 생성
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        # 데이터베이스 초기화
        self.db = StockDatabase(db_path)
        
        # 한국투자증권 API 초기화
        self.kis_api = KoreaInvestmentAPI()
        
        self.initialize_data()
        
        
    def initialize_data(self):
        """초기 데이터 로드"""
        logger.info("주식 데이터 초기화 중...")
        
        for code in STOCK_CODES:
            try:
                # 데이터베이스에서 데이터 로드 시도
                db_data = self.db.load_stock_data(code)
                
                if not db_data.empty:
                    logger.info(f"데이터베이스에서 종목코드 {code} 데이터를 로드했습니다: {len(db_data)} 행")
                    self.data[code] = db_data
                    
                    # 마지막 업데이트 시간 가져오기
                    last_update_time = self.db.get_last_update_time(code)
                    if last_update_time:
                        self.last_update[code] = last_update_time
                    else:
                        self.last_update[code] = datetime.now()
                    
                    continue  # 데이터베이스에서 로드 성공했으므로 다음 종목으로
                
                # 데이터베이스에 데이터가 없는 경우 API로 가져오기
                # 최근 100일 데이터 가져오기
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)
                
                # 한국투자증권 API로 데이터 가져오기
                start_date_str = start_date.strftime('%Y%m%d')
                end_date_str = end_date.strftime('%Y%m%d')
                
                logger.info(f"한국투자증권 API로 {code} 데이터를 가져오는 중...")
                df = self.kis_api.get_stock_ohlcv(code, start_date_str, end_date_str)
                
                # 한국투자증권 API로 데이터를 가져오지 못한 경우 pykrx 사용
                if df.empty:
                    logger.info(f"한국투자증권 API에서 {code} 데이터를 찾을 수 없어 pykrx 사용 시도 중...")
                    
                    df = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, code)
                    
                    # 컬럼명 변경하여 형식 맞추기
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
                
                if df.empty:
                    logger.warning(f"종목코드 {code}에 대한 실제 데이터를 가져올 수 없습니다. 샘플 데이터를 사용합니다.")
                else:
                    # 데이터베이스에 저장
                    self.db.save_stock_data(code, df)
                    logger.info(f"종목코드 {code} 데이터를 데이터베이스에 저장했습니다.")
                
                # 데이터 저장
                self.data[code] = df
                self.last_update[code] = datetime.now()
                
                logger.info(f"종목코드 {code} 데이터 로드 완료: {len(df)} 행")
                
            except Exception as e:
                logger.error(f"종목코드 {code} 데이터 로드 중 오류 발생: {str(e)}")
        
        # 모든 종목에 대해 데이터를 가져오지 못한 경우
        if all(code not in self.data or self.data[code].empty for code in STOCK_CODES):
            logger.warning("모든 종목에 대한 데이터를 가져오지 못했습니다. 샘플 데이터를 생성합니다.")
            
            # 샘플 데이터 생성
            for code in STOCK_CODES:
                if code not in self.data or self.data[code].empty:
                    self.data[code] = self._generate_sample_data(code)
                    self.last_update[code] = datetime.now()
                    self.sample_data_loaded = True
        
        if self.sample_data_loaded:
            logger.info("주식 데이터 초기화 완료 (샘플 데이터 사용)")
        else:
            logger.info("주식 데이터 초기화 완료 (실제 데이터 사용)")
    
    
    def update_data(self, force=False):
        """데이터 업데이트"""
        current_time = datetime.now()
        
        for code in STOCK_CODES:
            # 마지막 업데이트 이후 UPDATE_INTERVAL 초가 지났거나 강제 업데이트인 경우
            if (code not in self.last_update or
                (current_time - self.last_update[code]).total_seconds() > UPDATE_INTERVAL or
                force):
                
                try:
                    # 실제 환경에서는 실시간 API를 사용하겠지만, 여기서는 최신 데이터를 가져오는 것으로 시뮬레이션
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=1)  # 최근 1일 데이터
                    
                    # 날짜 형식 변환
                    start_date_str = start_date.strftime('%Y%m%d')
                    end_date_str = end_date.strftime('%Y%m%d')
                    
                    # 한국투자증권 API로 데이터 가져오기
                    logger.info(f"한국투자증권 API로 {code} 최신 데이터를 가져오는 중...")
                    new_data = self.kis_api.get_stock_ohlcv(code, start_date_str, end_date_str)
                    
                    # 한국투자증권 API로 데이터를 가져오지 못한 경우 pykrx 사용
                    if new_data.empty:
                        logger.info(f"한국투자증권 API에서 {code} 데이터를 찾을 수 없어 pykrx 사용 시도 중...")
                        
                        new_data = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, code)
                        
                        # 컬럼명 변경
                        if not new_data.empty:
                            new_data = new_data.rename(columns={
                                '시가': 'Open',
                                '고가': 'High',
                                '저가': 'Low',
                                '종가': 'Close',
                                '거래량': 'Volume'
                            })
                            
                            # Change 컬럼 추가
                            new_data['Change'] = new_data['Close'].pct_change()
                    
                    if not new_data.empty and code in self.data:
                        # NaN 값 처리
                        new_data = self._clean_data(new_data)
                        
                        # 기존 데이터와 새 데이터 병합 (중복 제거)
                        self.data[code] = pd.concat([self.data[code], new_data]).drop_duplicates()
                        self.last_update[code] = current_time
                        
                        # 데이터베이스에 업데이트된 데이터 저장
                        self.db.save_stock_data(code, self.data[code])
                        
                        logger.info(f"종목코드 {code} 데이터 업데이트 완료 및 데이터베이스 저장")
                    
                except Exception as e:
                    logger.error(f"종목코드 {code} 데이터 업데이트 중 오류 발생: {str(e)}")
    
    def get_latest_price(self, code):
        """최신 가격 정보 반환"""
        try:
            # 한국투자증권 API로 현재가 조회 시도
            current_price = self.kis_api.get_stock_current_price(code)
            if current_price:
                logger.info(f"한국투자증권 API에서 {code} 현재가를 가져왔습니다.")
                return current_price
        except Exception as e:
            logger.error(f"한국투자증권 API에서 {code} 현재가 조회 중 오류 발생: {str(e)}")
        
        # API 조회 실패 시 저장된 데이터에서 최신 가격 반환
        if code in self.data and not self.data[code].empty:
            # 원본 데이터 가져오기
            latest_data = self.data[code].iloc[-1].copy()
            
            # 다중 인덱스 처리
            def get_value(series, column_name):
                # Series가 MultiIndex를 가진 경우
                if isinstance(series.index, pd.MultiIndex):
                    # 해당 컬럼 이름으로 시작하는 첫 번째 인덱스 찾기
                    matching_indices = [idx for idx in series.index if idx[0] == column_name]
                    if matching_indices:
                        return series[matching_indices[0]]
                    return None
                # 일반 인덱스를 사용하는 경우
                else:
                    return series.get(column_name)
            
            result = {}
            
            for price_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                value = get_value(latest_data, price_col)
                
                # None 값을 기본값으로 대체
                if value is None:
                    if price_col in ['Open', 'High', 'Low', 'Close']:
                        # 가격 데이터의 경우 마지막 유효한 값 또는 0 사용
                        value = 0
                    elif price_col == 'Volume':
                        value = 0
                    elif price_col == 'Change':
                        value = 0
                
                result[price_col] = value
            
            return result
        return None
    
    # _get_price_conversion_factor 메서드는 한국투자증권 API에서 이미 원화 단위로 데이터를 제공하므로 제거
    
    def get_historical_data(self, code, days=30):
        """과거 데이터 반환"""
        if code in self.data and not self.data[code].empty:
            return self.data[code].tail(days)
        return None
    
    def simulate_realtime_data(self):
        """실시간 데이터 시뮬레이션 (테스트용)"""
        logger.info("실시간 데이터 시뮬레이션 시작")
        
        for code in STOCK_CODES:
            try:
                # 한국투자증권 API로 현재가 조회 시도
                current_price_info = self.kis_api.get_stock_current_price(code)
                
                if current_price_info:
                    logger.info(f"한국투자증권 API에서 {code} 현재가를 가져왔습니다.")
                    
                    # 현재 시간
                    current_time = datetime.now()
                    
                    # 새로운 행 생성
                    new_row = pd.DataFrame({
                        'Close': [current_price_info['Close']],
                        'High': [current_price_info['High']],
                        'Low': [current_price_info['Low']],
                        'Open': [current_price_info['Open']],
                        'Volume': [current_price_info['Volume']],
                        'Change': [current_price_info['Change']]
                    }, index=[current_time])
                    
                    # 데이터 추가 (기존 데이터가 있는 경우)
                    if code in self.data and not self.data[code].empty:
                        # 데이터프레임이 다중 인덱스인지 확인
                        if isinstance(self.data[code].columns, pd.MultiIndex):
                            # 다중 인덱스 컬럼 구조 유지
                            stock_columns = [(col, code) for col in new_row.columns]
                            new_row.columns = pd.MultiIndex.from_tuples(stock_columns)
                        
                        # 데이터 추가
                        self.data[code] = pd.concat([self.data[code], new_row])
                    else:
                        # 새로운 데이터 생성
                        self.data[code] = new_row
                    
                    # 마지막 업데이트 시간 갱신
                    self.last_update[code] = current_time
                    
                    # 주기적으로 데이터베이스에 저장 (매 10회 시뮬레이션마다)
                    if np.random.randint(0, 10) == 0:
                        self.db.save_stock_data(code, self.data[code])
                        logger.info(f"종목코드 {code} 실시간 데이터를 데이터베이스에 저장했습니다.")
                    
                    logger.info(f"종목코드 {code} 실시간 데이터 추가: 가격 {current_price_info['Close']:.2f} (변동률: {current_price_info['Change']*100:.2f}%)")
                    continue
            except Exception as e:
                logger.error(f"한국투자증권 API에서 {code} 현재가 조회 중 오류 발생: {str(e)}")
            
            # API 조회 실패 시 시뮬레이션 데이터 생성
            if code not in self.data or self.data[code].empty:
                continue
                
            # 최신 데이터 가져오기
            latest_data = self.get_latest_price(code)
            if latest_data is None:
                continue
                
            # Series 객체를 float로 변환하여 단일 값 사용
            last_price = float(latest_data['Close'])
            last_volume = int(latest_data['Volume'])
            
            # 랜덤한 가격 변동 (-1% ~ +1%)
            change_pct = np.random.uniform(-0.01, 0.01)
            new_price = last_price * (1 + change_pct)
            
            # 현재 시간
            current_time = datetime.now()
            
            # 새로운 행 생성
            new_row = pd.DataFrame({
                'Close': [new_price],
                'High': [max(last_price, new_price)],
                'Low': [min(last_price, new_price)],
                'Open': [last_price],
                'Volume': [int(np.random.uniform(0.5, 1.5) * last_volume)],
                'Change': [change_pct]
            }, index=[current_time])
            
            # 데이터프레임이 다중 인덱스인지 확인
            if isinstance(self.data[code].columns, pd.MultiIndex):
                # 다중 인덱스 컬럼 구조 유지
                stock_columns = [(col, code) for col in new_row.columns]
                new_row.columns = pd.MultiIndex.from_tuples(stock_columns)
            
            # 데이터 추가
            self.data[code] = pd.concat([self.data[code], new_row])
            
            # 마지막 업데이트 시간 갱신
            self.last_update[code] = current_time
            
            # 주기적으로 데이터베이스에 저장 (매 10회 시뮬레이션마다)
            if np.random.randint(0, 10) == 0:
                self.db.save_stock_data(code, self.data[code])
                logger.info(f"종목코드 {code} 시뮬레이션 데이터를 데이터베이스에 저장했습니다.")
            
            logger.info(f"종목코드 {code} 시뮬레이션 데이터 추가: 가격 {new_price:.2f} (변동률: {change_pct*100:.2f}%)")
        
        return {code: self.get_latest_price(code) for code in STOCK_CODES if code in self.data}
    
    def _clean_data(self, df):
        """데이터 정제 - NaN 값 처리"""
        if df is None or df.empty:
            return df
        
        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"필수 컬럼 {col}이 데이터프레임에 없습니다.")
                return df
        
        # 중앙 집중식 결측치 처리 함수 사용
        df_clean = handle_missing_values(df, method='ffill_bfill', columns=required_columns)
        
        # 인덱스 정렬
        df_clean = df_clean.sort_index()
        
        # Change 컬럼 재계산
        if 'Change' in df_clean.columns:
            df_clean['Change'] = df_clean['Close'].pct_change().fillna(0)
        
        return df_clean
    
    def _generate_sample_data(self, code):
        """샘플 주가 데이터 생성 (백테스팅 및 테스트용)"""
        logger.info(f"종목코드 {code}에 대한 샘플 데이터 생성 중...")
        
        # 샘플 데이터 기간 설정 (1년)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 날짜 범위 생성 (주말 제외)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 초기 가격 설정 (종목코드에 따라 다른 가격대 설정)
        if code == '005930':  # 삼성전자
            initial_price = 70000
        elif code == '000660':  # SK하이닉스
            initial_price = 120000
        elif code == '035720':  # 카카오
            initial_price = 50000
        elif code == '035420':  # NAVER
            initial_price = 200000
        elif code == '051910':  # LG화학
            initial_price = 500000
        elif code == '226330':  # 신테카 바이오
            initial_price = 8000
        else:
            initial_price = 50000  # 기본값
        
        # 랜덤 시드 설정 (종목코드를 시드로 사용하여 일관된 샘플 데이터 생성)
        seed = int(code) % 10000
        np.random.seed(seed)
        
        # 가격 변동 시뮬레이션 (랜덤 워크 + 추세)
        n_days = len(date_range)
        
        # 추세 성분 (상승 또는 하락 추세)
        trend = np.linspace(0, np.random.uniform(-0.3, 0.3), n_days)
        
        # 랜덤 워크 성분
        random_walk = np.random.normal(0, 0.015, n_days).cumsum()
        
        # 계절성 성분 (주기적 패턴)
        seasonality = 0.1 * np.sin(np.linspace(0, 4*np.pi, n_days))
        
        # 가격 변동률 계산
        price_changes = trend + random_walk + seasonality
        
        # 종가 계산
        close_prices = initial_price * (1 + price_changes)
        close_prices = np.maximum(close_prices, initial_price * 0.3)  # 가격이 너무 낮아지지 않도록 제한
        
        # 일중 변동폭 계산
        daily_volatility = np.random.uniform(0.01, 0.03, n_days)
        
        # OHLC 데이터 생성
        high_prices = close_prices * (1 + daily_volatility)
        low_prices = close_prices * (1 - daily_volatility)
        open_prices = np.zeros(n_days)
        
        # 시가는 전일 종가와 당일 종가 사이의 값으로 설정
        open_prices[0] = close_prices[0] * (1 - daily_volatility[0] * 0.5)
        for i in range(1, n_days):
            open_prices[i] = close_prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
            
            # 시가, 고가, 저가, 종가 관계 조정
            high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + np.random.uniform(0, daily_volatility[i]))
            low_prices[i] = min(open_prices[i], close_prices[i]) * (1 - np.random.uniform(0, daily_volatility[i]))
        
        # 거래량 생성 (가격 변동폭과 상관관계 있게 설정)
        base_volume = np.random.randint(100000, 1000000)
        volume = base_volume * (1 + np.abs(np.diff(np.append(open_prices[0], close_prices)) / close_prices) * 10)
        volume = volume.astype(int)
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=date_range)
        
        # Change 컬럼 추가
        df['Change'] = df['Close'].pct_change().fillna(0)
        
        logger.info(f"종목코드 {code}에 대한 샘플 데이터 생성 완료: {len(df)} 행")
        return df

# 테스트 코드
if __name__ == "__main__":
    stock_data = StockData()
    
    # 초기 데이터 출력
    for code in STOCK_CODES:
        if code in stock_data.data:
            print(f"종목코드: {code}")
            print(stock_data.data[code].tail())
            print("-" * 50)
    
    # 실시간 데이터 시뮬레이션
    for _ in range(5):
        latest_data = stock_data.simulate_realtime_data()
        time.sleep(1)
# 한국 주식 트레이딩 봇 설정 파일

# 모니터링할 주식 목록 (종목코드)
STOCK_CODES = [
    '226330',  # 신테카 바이오
]

# 데이터 업데이트 주기 (초)
UPDATE_INTERVAL = 60

import os
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발 환경용)
load_dotenv()

# 한국투자증권 API 설정
KIS_API = {
    'app_key': os.getenv('KIS_APP_KEY', ''),  # 실전 투자 앱키
    'app_secret': os.getenv('KIS_APP_SECRET', ''),  # 실전 투자 앱 시크릿
    'acc_no': os.getenv('KIS_ACC_NO', ''),  # 실전 투자 계좌번호
    'mock_app_key': os.getenv('KIS_MOCK_APP_KEY', ''),  # 모의 투자 앱키
    'mock_app_secret': os.getenv('KIS_MOCK_APP_SECRET', ''),  # 모의 투자 앱 시크릿
    'mock_acc_no': os.getenv('KIS_MOCK_ACC_NO', ''),  # 모의 투자 계좌번호
    'base_url': 'https://openapi.koreainvestment.com:9443',  # API 기본 URL
    'mock_url': 'https://openapivts.koreainvestment.com:29443',  # 모의투자 API URL
    'use_mock': os.getenv('KIS_USE_MOCK', 'True').lower() == 'true',  # 모의투자 사용 여부
    'token_path': os.getenv('KIS_TOKEN_PATH', 'data/kis_token.json'),  # 토큰 저장 경로
}

# 기술적 지표 설정
TECHNICAL_INDICATORS = {
    'SMA': {
        'short_period': 5,
        'long_period': 20
    },
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'MACD': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'Bollinger': {
        'period': 20,
        'std_dev': 2
    },
    'Stochastic': {
        'k_period': 14,
        'k_slowing_period': 3,
        'd_period': 3,
        'overbought': 80,
        'oversold': 20
    }
}

# 매매 신호 설정
TRADING_SIGNALS = {
    'sma_crossover': True,
    'rsi_bounds': True,
    'macd_crossover': True,
    'bollinger_breakout': True,
    'stochastic_crossover': True
}

# 백테스팅 설정
BACKTEST = {
    'start_date': '2023-01-01',
    'end_date': '2025-12-31',
    'initial_capital': 10000000,  # 1천만원
    'risk_management': {
        'stop_loss_pct': 3.0,     # 손절매 비율 (%)
        'take_profit_pct': 5.0,   # 익절매 비율 (%)
        'max_position_pct': 20.0, # 최대 포지션 비율 (%)
        'max_loss_pct': 10.0      # 최대 손실 비율 (%)
    }
}
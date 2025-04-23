"""
주식 데이터 관리 테스트 모듈
stock_data.py 파일의 StockData 클래스가 올바르게 동작하는지 확인합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def test_stock_data_imports():
    """StockData 클래스가 올바르게 임포트되는지 테스트합니다."""
    from stock_data import StockData
    assert callable(StockData), "StockData는 클래스여야 합니다"

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
def test_stock_data_init(mock_db, mock_api):
    """StockData 클래스 초기화를 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    mock_db_instance.load_stock_data.return_value = pd.DataFrame()
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    with patch('stock_data.StockData.initialize_data'):  # initialize_data 메서드 모킹
        stock_data = StockData()
        
        # 필수 속성 확인
        assert hasattr(stock_data, 'data'), "data 속성이 존재해야 합니다"
        assert hasattr(stock_data, 'last_update'), "last_update 속성이 존재해야 합니다"
        assert hasattr(stock_data, 'db'), "db 속성이 존재해야 합니다"
        assert hasattr(stock_data, 'kis_api'), "kis_api 속성이 존재해야 합니다"
        
        # 데이터베이스 및 API 초기화 확인
        mock_db.assert_called_once()
        mock_api.assert_called_once()

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_initialize_data_from_db(mock_db, mock_api):
    """데이터베이스에서 데이터를 로드하는 기능을 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    
    # 데이터베이스에서 로드할 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'Open': [70000, 70500, 71000],
        'High': [71000, 71500, 72000],
        'Low': [69000, 69500, 70000],
        'Close': [70500, 71000, 71500],
        'Volume': [1000000, 1100000, 1200000],
        'Change': [0.01, 0.007, 0.007]
    }, index=pd.date_range(start='2025-04-20', periods=3))
    
    mock_db_instance.load_stock_data.return_value = sample_data
    mock_db_instance.get_last_update_time.return_value = datetime.now()
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    stock_data = StockData()
    
    # 데이터가 올바르게 로드되었는지 확인
    assert '005930' in stock_data.data, "데이터가 로드되어야 합니다"
    assert not stock_data.data['005930'].empty, "데이터가 비어있지 않아야 합니다"
    assert len(stock_data.data['005930']) == 3, "데이터 길이가 3이어야 합니다"
    assert '005930' in stock_data.last_update, "마지막 업데이트 시간이 설정되어야 합니다"
    
    # 데이터베이스 호출 확인
    mock_db_instance.load_stock_data.assert_called_once_with('005930')
    mock_db_instance.get_last_update_time.assert_called_once_with('005930')

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_initialize_data_from_api(mock_db, mock_api):
    """API에서 데이터를 로드하는 기능을 테스트합니다."""
    # 데이터베이스 모킹 (빈 데이터 반환)
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    mock_db_instance.load_stock_data.return_value = pd.DataFrame()
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # API에서 로드할 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'Open': [70000, 70500, 71000],
        'High': [71000, 71500, 72000],
        'Low': [69000, 69500, 70000],
        'Close': [70500, 71000, 71500],
        'Volume': [1000000, 1100000, 1200000],
        'Change': [0.01, 0.007, 0.007]
    }, index=pd.date_range(start='2025-04-20', periods=3))
    
    mock_api_instance.get_stock_ohlcv.return_value = sample_data
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    stock_data = StockData()
    
    # 데이터가 올바르게 로드되었는지 확인
    assert '005930' in stock_data.data, "데이터가 로드되어야 합니다"
    assert not stock_data.data['005930'].empty, "데이터가 비어있지 않아야 합니다"
    assert len(stock_data.data['005930']) == 3, "데이터 길이가 3이어야 합니다"
    assert '005930' in stock_data.last_update, "마지막 업데이트 시간이 설정되어야 합니다"
    
    # API 호출 확인
    mock_api_instance.get_stock_ohlcv.assert_called_once()
    
    # 데이터베이스 저장 확인
    mock_db_instance.save_stock_data.assert_called_once()

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_update_data(mock_db, mock_api):
    """데이터 업데이트 기능을 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    
    # 기존 데이터 생성
    existing_data = pd.DataFrame({
        'Open': [70000, 70500, 71000],
        'High': [71000, 71500, 72000],
        'Low': [69000, 69500, 70000],
        'Close': [70500, 71000, 71500],
        'Volume': [1000000, 1100000, 1200000],
        'Change': [0.01, 0.007, 0.007]
    }, index=pd.date_range(start='2025-04-20', periods=3))
    
    mock_db_instance.load_stock_data.return_value = existing_data
    mock_db_instance.get_last_update_time.return_value = datetime.now() - timedelta(minutes=2)
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 새로운 데이터 생성
    new_data = pd.DataFrame({
        'Open': [71500],
        'High': [72000],
        'Low': [71000],
        'Close': [71800],
        'Volume': [1300000],
        'Change': [0.004]
    }, index=pd.date_range(start='2025-04-23', periods=1))
    
    mock_api_instance.get_stock_ohlcv.return_value = new_data
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    with patch('stock_data.UPDATE_INTERVAL', 60):  # 업데이트 주기를 60초로 설정
        stock_data = StockData()
        
        # 초기 데이터 확인
        assert len(stock_data.data['005930']) == 3, "초기 데이터 길이가 3이어야 합니다"
        
        # 데이터 업데이트
        stock_data.update_data(force=True)
        
        # 업데이트된 데이터 확인
        assert len(stock_data.data['005930']) == 4, "업데이트 후 데이터 길이가 4여야 합니다"
        
        # API 호출 확인
        assert mock_api_instance.get_stock_ohlcv.call_count == 2  # 초기화 + 업데이트
        
        # 데이터베이스 저장 확인
        assert mock_db_instance.save_stock_data.call_count == 2  # 초기화 + 업데이트

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_get_latest_price(mock_db, mock_api):
    """최신 가격 정보 조회 기능을 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    
    # 기존 데이터 생성
    existing_data = pd.DataFrame({
        'Open': [70000, 70500, 71000],
        'High': [71000, 71500, 72000],
        'Low': [69000, 69500, 70000],
        'Close': [70500, 71000, 71500],
        'Volume': [1000000, 1100000, 1200000],
        'Change': [0.01, 0.007, 0.007]
    }, index=pd.date_range(start='2025-04-20', periods=3))
    
    mock_db_instance.load_stock_data.return_value = existing_data
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 현재가 정보 생성
    current_price = {
        'Open': 71000,
        'High': 72000,
        'Low': 70500,
        'Close': 71800,
        'Volume': 1300000,
        'Change': 0.004
    }
    
    mock_api_instance.get_stock_current_price.return_value = current_price
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    stock_data = StockData()
    
    # 최신 가격 정보 조회
    latest_price = stock_data.get_latest_price('005930')
    
    # 결과 확인
    assert latest_price == current_price, "API에서 반환한 현재가 정보와 일치해야 합니다"
    
    # API 호출 확인
    mock_api_instance.get_stock_current_price.assert_called_once_with('005930')
    
    # API 오류 시 저장된 데이터에서 최신 가격 반환 테스트
    mock_api_instance.get_stock_current_price.side_effect = Exception("API 오류")
    
    latest_price = stock_data.get_latest_price('005930')
    
    # 결과 확인
    assert isinstance(latest_price, dict), "저장된 데이터에서 최신 가격 정보를 반환해야 합니다"
    assert 'Close' in latest_price, "Close 키가 존재해야 합니다"
    assert latest_price['Close'] == 71500, "마지막 종가가 반환되어야 합니다"

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_get_historical_data(mock_db, mock_api):
    """과거 데이터 조회 기능을 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    
    # 기존 데이터 생성 (30일)
    date_range = pd.date_range(end=datetime.now(), periods=30)
    data = {
        'Open': np.random.randint(70000, 72000, 30),
        'High': np.random.randint(71000, 73000, 30),
        'Low': np.random.randint(69000, 71000, 30),
        'Close': np.random.randint(70000, 72000, 30),
        'Volume': np.random.randint(1000000, 1500000, 30),
        'Change': np.random.uniform(-0.01, 0.01, 30)
    }
    existing_data = pd.DataFrame(data, index=date_range)
    
    mock_db_instance.load_stock_data.return_value = existing_data
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    stock_data = StockData()
    
    # 과거 데이터 조회 (기본 30일)
    historical_data = stock_data.get_historical_data('005930')
    
    # 결과 확인
    assert len(historical_data) == 30, "기본 30일 데이터가 반환되어야 합니다"
    
    # 과거 데이터 조회 (10일)
    historical_data = stock_data.get_historical_data('005930', days=10)
    
    # 결과 확인
    assert len(historical_data) == 10, "요청한 10일 데이터가 반환되어야 합니다"
    
    # 존재하지 않는 종목 코드 테스트
    historical_data = stock_data.get_historical_data('000000')
    
    # 결과 확인
    assert historical_data is None, "존재하지 않는 종목 코드는 None을 반환해야 합니다"

@patch('stock_data.KoreaInvestmentAPI')
@patch('stock_data.StockDatabase')
@patch('stock_data.STOCK_CODES', ['005930'])  # 삼성전자 코드만 사용
def test_clean_data(mock_db, mock_api):
    """데이터 정제 기능을 테스트합니다."""
    # 데이터베이스 모킹
    mock_db_instance = MagicMock()
    mock_db.return_value = mock_db_instance
    mock_db_instance.load_stock_data.return_value = pd.DataFrame()
    
    # API 모킹
    mock_api_instance = MagicMock()
    mock_api.return_value = mock_api_instance
    
    # 클래스 임포트 및 초기화
    from stock_data import StockData
    with patch('stock_data.StockData.initialize_data'):  # initialize_data 메서드 모킹
        stock_data = StockData()
        
        # 결측치가 있는 데이터 생성
        data_with_nan = pd.DataFrame({
            'Open': [70000, np.nan, 71000],
            'High': [71000, 71500, np.nan],
            'Low': [69000, 69500, 70000],
            'Close': [70500, 71000, 71500],
            'Volume': [1000000, np.nan, 1200000],
            'Change': [0.01, 0.007, 0.007]
        }, index=pd.date_range(start='2025-04-20', periods=3))
        
        # 데이터 정제
        clean_data = stock_data._clean_data(data_with_nan)
        
        # 결과 확인
        assert clean_data is not None, "정제된 데이터가 반환되어야 합니다"
        assert not clean_data.isnull().any().any(), "결측치가 없어야 합니다"
        assert clean_data['Open'][1] == 70000, "Open 결측치가 올바르게 채워져야 합니다"
        assert clean_data['High'][2] == 71500, "High 결측치가 올바르게 채워져야 합니다"
        assert clean_data['Volume'][1] == 1000000, "Volume 결측치가 올바르게 채워져야 합니다"
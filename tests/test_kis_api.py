"""
한국투자증권 API 테스트 모듈
kis_api.py 파일의 KoreaInvestmentAPI 클래스가 올바르게 동작하는지 확인합니다.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import json
import os

def test_kis_api_imports():
    """KoreaInvestmentAPI 클래스가 올바르게 임포트되는지 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    assert callable(KoreaInvestmentAPI), "KoreaInvestmentAPI는 클래스여야 합니다"

@patch('kis_api.KoreaInvestmentAPI._load_or_issue_token')
def test_kis_api_init(mock_load_token):
    """KoreaInvestmentAPI 클래스 초기화를 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    
    # 토큰 로드 함수 모킹
    mock_load_token.return_value = None
    
    # 클래스 초기화
    api = KoreaInvestmentAPI()
    
    # 필수 속성 확인
    assert hasattr(api, 'app_key'), "app_key 속성이 존재해야 합니다"
    assert hasattr(api, 'app_secret'), "app_secret 속성이 존재해야 합니다"
    assert hasattr(api, 'acc_no'), "acc_no 속성이 존재해야 합니다"
    assert hasattr(api, 'base_url'), "base_url 속성이 존재해야 합니다"
    assert hasattr(api, 'token_path'), "token_path 속성이 존재해야 합니다"
    
    # 토큰 로드 함수 호출 확인
    mock_load_token.assert_called_once()

@patch('kis_api.KoreaInvestmentAPI._load_or_issue_token')
def test_kis_api_methods(mock_load_token):
    """KoreaInvestmentAPI 클래스의 메서드들이 존재하는지 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    
    # 토큰 로드 함수 모킹
    mock_load_token.return_value = None
    
    # 클래스 초기화
    api = KoreaInvestmentAPI()
    
    # 필수 메서드 확인
    assert callable(getattr(api, '_load_or_issue_token', None)), "_load_or_issue_token 메서드가 존재해야 합니다"
    assert callable(getattr(api, '_issue_token', None)), "_issue_token 메서드가 존재해야 합니다"
    assert callable(getattr(api, '_check_token', None)), "_check_token 메서드가 존재해야 합니다"
    assert callable(getattr(api, 'get_stock_ohlcv', None)), "get_stock_ohlcv 메서드가 존재해야 합니다"
    assert callable(getattr(api, 'get_stock_current_price', None)), "get_stock_current_price 메서드가 존재해야 합니다"
    assert callable(getattr(api, 'get_account_balance', None)), "get_account_balance 메서드가 존재해야 합니다"
    assert callable(getattr(api, 'place_order', None)), "place_order 메서드가 존재해야 합니다"
    assert callable(getattr(api, 'get_order_history', None)), "get_order_history 메서드가 존재해야 합니다"

@patch('kis_api.KoreaInvestmentAPI._load_or_issue_token')
@patch('kis_api.KoreaInvestmentAPI._check_token')
@patch('requests.get')
def test_get_stock_ohlcv(mock_get, mock_check_token, mock_load_token):
    """get_stock_ohlcv 메서드를 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    
    # 토큰 로드 함수 모킹
    mock_load_token.return_value = None
    mock_check_token.return_value = None
    
    # API 응답 모킹
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        'output': {},
        'output1': [
            {
                'stck_bsop_date': '20250423',
                'stck_oprc': '70000',
                'stck_hgpr': '71000',
                'stck_lwpr': '69000',
                'stck_clpr': '70500',
                'acml_vol': '10000000'
            },
            {
                'stck_bsop_date': '20250422',
                'stck_oprc': '69000',
                'stck_hgpr': '70000',
                'stck_lwpr': '68500',
                'stck_clpr': '70000',
                'acml_vol': '9000000'
            }
        ]
    }
    mock_get.return_value = mock_response
    
    # 클래스 초기화
    api = KoreaInvestmentAPI()
    
    # 메서드 호출
    df = api.get_stock_ohlcv('005930', period=30)
    
    # 결과 확인
    assert isinstance(df, pd.DataFrame), "반환값은 DataFrame이어야 합니다"
    assert not df.empty, "DataFrame은 비어있지 않아야 합니다"
    assert len(df) == 2, "DataFrame은 2개의 행을 가져야 합니다"
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Change'], "DataFrame은 올바른 컬럼을 가져야 합니다"
    
    # API 호출 확인
    mock_check_token.assert_called_once()
    mock_get.assert_called_once()

@patch('kis_api.KoreaInvestmentAPI._load_or_issue_token')
@patch('kis_api.KoreaInvestmentAPI._check_token')
@patch('requests.get')
def test_get_stock_current_price(mock_get, mock_check_token, mock_load_token):
    """get_stock_current_price 메서드를 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    
    # 토큰 로드 함수 모킹
    mock_load_token.return_value = None
    mock_check_token.return_value = None
    
    # API 응답 모킹
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        'output': {
            'stck_oprc': '70000',
            'stck_hgpr': '71000',
            'stck_lwpr': '69000',
            'stck_prpr': '70500',
            'acml_vol': '10000000',
            'prdy_ctrt': '1.5'
        }
    }
    mock_get.return_value = mock_response
    
    # 클래스 초기화
    api = KoreaInvestmentAPI()
    
    # 메서드 호출
    price_info = api.get_stock_current_price('005930')
    
    # 결과 확인
    assert isinstance(price_info, dict), "반환값은 딕셔너리여야 합니다"
    assert 'Open' in price_info, "Open 키가 존재해야 합니다"
    assert 'High' in price_info, "High 키가 존재해야 합니다"
    assert 'Low' in price_info, "Low 키가 존재해야 합니다"
    assert 'Close' in price_info, "Close 키가 존재해야 합니다"
    assert 'Volume' in price_info, "Volume 키가 존재해야 합니다"
    assert 'Change' in price_info, "Change 키가 존재해야 합니다"
    
    # API 호출 확인
    mock_check_token.assert_called_once()
    mock_get.assert_called_once()

@patch('kis_api.KoreaInvestmentAPI._load_or_issue_token')
@patch('kis_api.KoreaInvestmentAPI._issue_token')
def test_check_token(mock_issue_token, mock_load_token):
    """_check_token 메서드를 테스트합니다."""
    from kis_api import KoreaInvestmentAPI
    
    # 토큰 로드 함수 모킹
    mock_load_token.return_value = None
    mock_issue_token.return_value = None
    
    # 클래스 초기화
    api = KoreaInvestmentAPI()
    
    # 토큰이 없는 경우
    api.access_token = None
    api._check_token()
    mock_issue_token.assert_called_once()
    mock_issue_token.reset_mock()
    
    # 토큰이 만료된 경우
    api.access_token = "test_token"
    api.token_expired_at = datetime.now() - timedelta(hours=1)
    api._check_token()
    mock_issue_token.assert_called_once()
    mock_issue_token.reset_mock()
    
    # 토큰이 유효한 경우
    api.access_token = "test_token"
    api.token_expired_at = datetime.now() + timedelta(hours=1)
    api._check_token()
    mock_issue_token.assert_not_called()
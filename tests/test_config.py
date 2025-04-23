"""
설정 파일 테스트 모듈
config.py 파일의 설정이 올바르게 로드되고 형식이 맞는지 확인합니다.
"""

def test_config_imports():
    """설정 파일이 올바르게 임포트되는지 테스트합니다."""
    import config
    assert hasattr(config, 'STOCK_CODES'), "STOCK_CODES 설정이 존재해야 합니다"
    assert hasattr(config, 'UPDATE_INTERVAL'), "UPDATE_INTERVAL 설정이 존재해야 합니다"
    assert hasattr(config, 'KIS_API'), "KIS_API 설정이 존재해야 합니다"
    assert hasattr(config, 'TECHNICAL_INDICATORS'), "TECHNICAL_INDICATORS 설정이 존재해야 합니다"
    assert hasattr(config, 'TRADING_SIGNALS'), "TRADING_SIGNALS 설정이 존재해야 합니다"
    assert hasattr(config, 'BACKTEST'), "BACKTEST 설정이 존재해야 합니다"

def test_stock_codes():
    """주식 코드 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.STOCK_CODES, list), "STOCK_CODES는 리스트 형식이어야 합니다"
    for code in config.STOCK_CODES:
        assert isinstance(code, str), "주식 코드는 문자열이어야 합니다"
        assert code.isdigit(), "주식 코드는 숫자로만 구성되어야 합니다"

def test_update_interval():
    """업데이트 주기 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.UPDATE_INTERVAL, int), "UPDATE_INTERVAL은 정수여야 합니다"
    assert config.UPDATE_INTERVAL > 0, "UPDATE_INTERVAL은 양수여야 합니다"

def test_kis_api_config():
    """한국투자증권 API 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.KIS_API, dict), "KIS_API는 딕셔너리 형식이어야 합니다"
    required_keys = ['app_key', 'app_secret', 'acc_no', 'base_url', 'use_mock']
    for key in required_keys:
        assert key in config.KIS_API, f"KIS_API에 '{key}' 키가 존재해야 합니다"

def test_technical_indicators():
    """기술적 지표 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.TECHNICAL_INDICATORS, dict), "TECHNICAL_INDICATORS는 딕셔너리 형식이어야 합니다"
    expected_indicators = ['SMA', 'RSI', 'MACD', 'Bollinger', 'Stochastic']
    for indicator in expected_indicators:
        assert indicator in config.TECHNICAL_INDICATORS, f"TECHNICAL_INDICATORS에 '{indicator}' 키가 존재해야 합니다"

def test_trading_signals():
    """매매 신호 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.TRADING_SIGNALS, dict), "TRADING_SIGNALS는 딕셔너리 형식이어야 합니다"
    expected_signals = ['sma_crossover', 'rsi_bounds', 'macd_crossover', 'bollinger_breakout', 'stochastic_crossover']
    for signal in expected_signals:
        assert signal in config.TRADING_SIGNALS, f"TRADING_SIGNALS에 '{signal}' 키가 존재해야 합니다"
        assert isinstance(config.TRADING_SIGNALS[signal], bool), f"'{signal}' 값은 불리언이어야 합니다"

def test_backtest_config():
    """백테스팅 설정이 올바른 형식인지 테스트합니다."""
    import config
    assert isinstance(config.BACKTEST, dict), "BACKTEST는 딕셔너리 형식이어야 합니다"
    required_keys = ['start_date', 'end_date', 'initial_capital', 'risk_management']
    for key in required_keys:
        assert key in config.BACKTEST, f"BACKTEST에 '{key}' 키가 존재해야 합니다"
    
    assert isinstance(config.BACKTEST['initial_capital'], int), "initial_capital은 정수여야 합니다"
    assert config.BACKTEST['initial_capital'] > 0, "initial_capital은 양수여야 합니다"
    
    assert isinstance(config.BACKTEST['risk_management'], dict), "risk_management는 딕셔너리 형식이어야 합니다"
    risk_keys = ['stop_loss_pct', 'take_profit_pct', 'max_position_pct', 'max_loss_pct']
    for key in risk_keys:
        assert key in config.BACKTEST['risk_management'], f"risk_management에 '{key}' 키가 존재해야 합니다"
        assert isinstance(config.BACKTEST['risk_management'][key], float), f"'{key}' 값은 실수여야 합니다"
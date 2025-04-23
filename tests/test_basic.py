"""
기본 테스트 모듈
모듈 임포트와 기본 기능이 정상적으로 작동하는지 확인합니다.
"""

def test_imports():
    """모든 주요 모듈이 정상적으로 임포트되는지 테스트합니다."""
    try:
        import app
        import config
        import database
        import kis_api
        import main
        import stock_data
        import technical_indicators
        import trading_signals
        import utils
        assert True  # 임포트가 성공하면 테스트 통과
    except ImportError as e:
        assert False, f"모듈 임포트 실패: {e}"

def test_environment():
    """기본 환경이 정상적으로 설정되어 있는지 테스트합니다."""
    import os
    import sys
    
    # 현재 작업 디렉토리가 올바른지 확인
    assert os.path.exists("requirements.txt"), "requirements.txt 파일이 존재해야 합니다"
    
    # Python 버전 확인
    assert sys.version_info.major == 3, "Python 3가 필요합니다"
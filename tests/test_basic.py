"""
기본 테스트 모듈
모듈 임포트와 기본 기능이 정상적으로 작동하는지 확인합니다.
"""

import unittest.mock
import os
import sys

def test_module_files_exist():
    """모든 주요 모듈 파일이 존재하는지 테스트합니다."""
    module_files = [
        "app.py",
        "config.py",
        "database.py",
        "kis_api.py",
        "main.py",
        "stock_data.py",
        "technical_indicators.py",
        "trading_signals.py",
        "utils.py"
    ]
    
    for module_file in module_files:
        assert os.path.exists(module_file), f"{module_file} 파일이 존재해야 합니다"

# API 호출 모킹을 위한 패치 설정
@unittest.mock.patch('requests.post')
@unittest.mock.patch('requests.get')
def test_imports(mock_get, mock_post):
    """모든 주요 모듈이 정상적으로 임포트되는지 테스트합니다."""
    # API 응답 모킹
    mock_response = unittest.mock.MagicMock()
    mock_response.json.return_value = {'access_token': 'test_token'}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    mock_get.return_value = mock_response
    
    try:
        # 모듈 임포트
        import config
        import database
        import kis_api
        import utils
        import technical_indicators
        import trading_signals
        import stock_data
        
        # app과 main 모듈은 직접적인 API 호출이 있을 수 있으므로 마지막에 테스트
        import main
        import app
        
        assert True  # 임포트가 성공하면 테스트 통과
    except ImportError as e:
        assert False, f"모듈 임포트 실패: {e}"
    except Exception as e:
        assert False, f"모듈 임포트 중 오류 발생: {e}"

def test_environment():
    """기본 환경이 정상적으로 설정되어 있는지 테스트합니다."""
    import os
    import sys
    
    # 현재 작업 디렉토리가 올바른지 확인
    assert os.path.exists("requirements.txt"), "requirements.txt 파일이 존재해야 합니다"
    
    # Python 버전 확인
    assert sys.version_info.major == 3, "Python 3가 필요합니다"
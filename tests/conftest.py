"""
pytest 설정 파일
테스트 환경 설정 및 공통 기능을 제공합니다.
"""

import os
import sys

# 프로젝트 루트 디렉토리를 모듈 검색 경로에 추가
# 이렇게 하면 tests 디렉토리에서 실행하더라도 프로젝트 모듈을 임포트할 수 있습니다.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 테스트 실행 시 경로 정보 출력 (디버깅용)
print(f"프로젝트 루트 디렉토리: {project_root}")
print(f"Python 모듈 검색 경로: {sys.path}")
"""
유틸리티 함수 테스트 모듈
utils.py 파일의 함수들이 올바르게 동작하는지 확인합니다.
"""

import pandas as pd
import numpy as np
import pytest

def test_utils_imports():
    """유틸리티 모듈이 올바르게 임포트되는지 테스트합니다."""
    import utils
    assert hasattr(utils, 'handle_missing_values'), "handle_missing_values 함수가 존재해야 합니다"
    assert hasattr(utils, 'logger'), "logger가 존재해야 합니다"

def test_handle_missing_values_empty_df():
    """빈 데이터프레임에 대한 처리를 테스트합니다."""
    import utils
    
    # 빈 데이터프레임
    df = pd.DataFrame()
    result = utils.handle_missing_values(df)
    assert result.empty, "빈 데이터프레임은 빈 데이터프레임을 반환해야 합니다"
    
    # None 입력
    result = utils.handle_missing_values(None)
    assert result is None, "None 입력은 None을 반환해야 합니다"

def test_handle_missing_values_no_missing():
    """결측치가 없는 데이터프레임에 대한 처리를 테스트합니다."""
    import utils
    
    # 결측치가 없는 데이터프레임
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    result = utils.handle_missing_values(df)
    pd.testing.assert_frame_equal(result, df), "결측치가 없는 경우 원본 데이터프레임을 반환해야 합니다"

def test_handle_missing_values_ffill_bfill():
    """ffill_bfill 방식의 결측치 처리를 테스트합니다."""
    import utils
    
    # 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, np.nan]
    })
    
    expected = pd.DataFrame({
        'A': [1, 1, 3, 3, 5],
        'B': [10, 20, 20, 40, 40]
    })
    
    result = utils.handle_missing_values(df, method='ffill_bfill')
    pd.testing.assert_frame_equal(result, expected), "ffill_bfill 방식이 올바르게 적용되어야 합니다"

def test_handle_missing_values_interpolate():
    """interpolate 방식의 결측치 처리를 테스트합니다."""
    import utils
    
    # 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, np.nan]
    })
    
    expected = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],  # 선형 보간
        'B': [10, 20, 30, 40, 40]  # 선형 보간 후 bfill
    })
    
    result = utils.handle_missing_values(df, method='interpolate')
    pd.testing.assert_frame_equal(result, expected), "interpolate 방식이 올바르게 적용되어야 합니다"

def test_handle_missing_values_zero():
    """zero 방식의 결측치 처리를 테스트합니다."""
    import utils
    
    # 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, np.nan]
    })
    
    expected = pd.DataFrame({
        'A': [1, 0, 3, 0, 5],
        'B': [10, 20, 0, 40, 0]
    })
    
    result = utils.handle_missing_values(df, method='zero')
    pd.testing.assert_frame_equal(result, expected), "zero 방식이 올바르게 적용되어야 합니다"

def test_handle_missing_values_drop():
    """drop 방식의 결측치 처리를 테스트합니다."""
    import utils
    
    # 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, 50]
    })
    
    # A와 B 모두 결측치가 없는 행만 남음
    expected = pd.DataFrame({
        'A': [1, 5],
        'B': [10, 50]
    }, index=[0, 4])
    
    result = utils.handle_missing_values(df, method='drop')
    pd.testing.assert_frame_equal(result, expected), "drop 방식이 올바르게 적용되어야 합니다"

def test_handle_missing_values_specific_columns():
    """특정 컬럼만 처리하는 기능을 테스트합니다."""
    import utils
    
    # 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [10, 20, np.nan, 40, np.nan],
        'C': [100, np.nan, 300, np.nan, 500]
    })
    
    # A와 B만 처리, C는 그대로
    expected = pd.DataFrame({
        'A': [1, 1, 3, 3, 5],
        'B': [10, 20, 20, 40, 40],
        'C': [100, np.nan, 300, np.nan, 500]
    })
    
    result = utils.handle_missing_values(df, method='ffill_bfill', columns=['A', 'B'])
    pd.testing.assert_frame_equal(result, expected), "특정 컬럼만 처리되어야 합니다"

def test_handle_missing_values_max_gap():
    """최대 간격 제한이 있는 보간을 테스트합니다."""
    import utils
    
    # 연속된 결측치가 있는 데이터프레임
    df = pd.DataFrame({
        'A': [1, np.nan, np.nan, np.nan, 5],
        'B': [10, np.nan, np.nan, 40, 50]
    })
    
    # max_gap=1이면 1개까지만 보간, 나머지는 ffill_bfill
    expected = pd.DataFrame({
        'A': [1, 1, 1, 1, 5],  # 간격이 3이라 모두 ffill_bfill
        'B': [10, 10, 40, 40, 50]  # 첫 번째 NaN만 보간, 두 번째는 ffill_bfill
    })
    
    result = utils.handle_missing_values(df, method='interpolate', max_gap=1)
    pd.testing.assert_frame_equal(result, expected), "최대 간격 제한이 올바르게 적용되어야 합니다"
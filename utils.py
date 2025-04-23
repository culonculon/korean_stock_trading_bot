import pandas as pd
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('utils')

def handle_missing_values(df, method='ffill_bfill', columns=None, max_gap=None):
    """
    중앙 집중식 결측치 처리 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        결측치를 처리할 데이터프레임
    method : str
        결측치 처리 방법 ('ffill_bfill', 'interpolate', 'drop', 'zero')
    columns : list
        처리할 컬럼 목록 (None이면 모든 컬럼)
    max_gap : int
        보간할 최대 결측치 간격 (None이면 제한 없음)
        
    Returns:
    --------
    pandas.DataFrame
        결측치가 처리된 데이터프레임
    """
    if df is None or df.empty:
        return df
        
    # 처리할 컬럼 결정
    if columns is None:
        # 다중 인덱스 처리
        if isinstance(df.columns, pd.MultiIndex):
            # 첫 번째 레벨의 컬럼 이름만 추출
            columns = list(set([col[0] for col in df.columns]))
        else:
            columns = df.columns.tolist()
    
    # 결측치 개수 확인
    missing_before = df[columns].isnull().sum()
    if missing_before.sum() > 0:
        logger.info(f"결측치 처리 전: {missing_before}")
        
        # 데이터프레임 복사
        df_clean = df.copy()
        
        # 결측치 처리 방법에 따라 처리
        if method == 'ffill_bfill':
            # 전진 채우기 후 후진 채우기
            for col in columns:
                if isinstance(df_clean.columns, pd.MultiIndex):
                    # 다중 인덱스인 경우 해당 컬럼 이름으로 시작하는 모든 컬럼 처리
                    matching_cols = [c for c in df_clean.columns if c[0] == col]
                    for mc in matching_cols:
                        df_clean[mc] = df_clean[mc].fillna(method='ffill').fillna(method='bfill')
                else:
                    # 일반 인덱스인 경우
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'interpolate':
            # 선형 보간법
            for col in columns:
                if isinstance(df_clean.columns, pd.MultiIndex):
                    matching_cols = [c for c in df_clean.columns if c[0] == col]
                    for mc in matching_cols:
                        if max_gap:
                            # 최대 간격 이내의 결측치만 보간
                            mask = df_clean[mc].isnull()
                            grp = ((mask != mask.shift(1)) | (mask.shift(1) == False)).cumsum()
                            cnt = mask.groupby(grp).transform('sum')
                            to_fill = mask & (cnt <= max_gap)
                            df_clean.loc[to_fill, mc] = df_clean.loc[to_fill, mc].interpolate(method='linear')
                            # 나머지는 ffill_bfill로 처리
                            df_clean[mc] = df_clean[mc].fillna(method='ffill').fillna(method='bfill')
                        else:
                            df_clean[mc] = df_clean[mc].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                else:
                    if col in df_clean.columns:
                        if max_gap:
                            mask = df_clean[col].isnull()
                            grp = ((mask != mask.shift(1)) | (mask.shift(1) == False)).cumsum()
                            cnt = mask.groupby(grp).transform('sum')
                            to_fill = mask & (cnt <= max_gap)
                            df_clean.loc[to_fill, col] = df_clean.loc[to_fill, col].interpolate(method='linear')
                            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                        else:
                            df_clean[col] = df_clean[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'drop':
            # 결측치가 있는 행 제거
            if isinstance(df_clean.columns, pd.MultiIndex):
                # 다중 인덱스인 경우 해당 컬럼들 찾기
                cols_to_check = []
                for col in columns:
                    matching_cols = [c for c in df_clean.columns if c[0] == col]
                    cols_to_check.extend(matching_cols)
                df_clean = df_clean.dropna(subset=cols_to_check)
            else:
                # 일반 인덱스인 경우
                cols_to_check = [col for col in columns if col in df_clean.columns]
                df_clean = df_clean.dropna(subset=cols_to_check)
            
            # 결측치가 너무 많이 제거되었는지 확인
            if len(df_clean) < len(df) * 0.7:  # 30% 이상 제거되면 경고
                logger.warning(f"결측치 제거로 인해 데이터의 {100 - len(df_clean)/len(df)*100:.1f}%가 손실되었습니다.")
        
        elif method == 'zero':
            # 0으로 채우기
            for col in columns:
                if isinstance(df_clean.columns, pd.MultiIndex):
                    matching_cols = [c for c in df_clean.columns if c[0] == col]
                    for mc in matching_cols:
                        df_clean[mc] = df_clean[mc].fillna(0)
                else:
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].fillna(0)
        
        # 결측치 처리 후 확인
        missing_after = df_clean[columns].isnull().sum()
        if missing_after.sum() > 0:
            logger.warning(f"결측치 처리 후에도 남아있는 결측치: {missing_after}")
        else:
            logger.info("모든 결측치가 처리되었습니다.")
            
        return df_clean
    else:
        logger.info("결측치가 없습니다.")
        return df
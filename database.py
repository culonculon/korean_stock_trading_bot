import sqlite3
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database')

class StockDatabase:
    """주식 데이터를 저장하고 관리하는 데이터베이스 클래스"""
    
    def __init__(self, db_path='stock_data.db'):
        """
        데이터베이스 초기화
        
        Parameters:
        -----------
        db_path : str
            데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._create_database()
        
    def _create_database(self):
        """데이터베이스 및 테이블 생성"""
        try:
            # 데이터베이스 디렉토리 확인 및 생성
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # 데이터베이스 연결
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 주식 데이터 테이블 생성
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                change REAL,
                UNIQUE(code, date)
            )
            ''')
            
            # 마지막 업데이트 시간 테이블 생성
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS last_update (
                code TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            raise
    
    def save_stock_data(self, code, df):
        """
        주식 데이터를 데이터베이스에 저장
        
        Parameters:
        -----------
        code : str
            종목 코드
        df : pandas.DataFrame
            저장할 주식 데이터
        """
        if df is None or df.empty:
            logger.warning(f"저장할 데이터가 없습니다: {code}")
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 데이터프레임 준비
            df_to_save = df.copy()
            
            # 인덱스가 날짜인지 확인하고 처리
            if isinstance(df_to_save.index, pd.DatetimeIndex):
                df_to_save = df_to_save.reset_index()
                df_to_save.rename(columns={'index': 'date'}, inplace=True)
                df_to_save['date'] = df_to_save['date'].dt.strftime('%Y-%m-%d')
            
            # 필요한 컬럼 확인 및 추가
            required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
            for col in required_columns:
                if col not in df_to_save.columns:
                    if col == 'Change':
                        # Change 컬럼이 없으면 Close 컬럼으로부터 계산
                        if 'Close' in df_to_save.columns:
                            df_to_save['Change'] = df_to_save['Close'].pct_change().fillna(0)
                        else:
                            df_to_save['Change'] = 0
                    else:
                        df_to_save[col] = None
            
            # 컬럼명 소문자로 변경
            df_to_save.columns = [col.lower() if col != 'date' else col for col in df_to_save.columns]
            
            # code 컬럼 추가
            df_to_save['code'] = code
            
            # 데이터베이스에 저장
            df_to_save[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'change']].to_sql(
                'stock_data', 
                conn, 
                if_exists='replace', 
                index=False
            )
            
            # 마지막 업데이트 시간 저장
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO last_update (code, timestamp) VALUES (?, ?)",
                (code, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"종목코드 {code} 데이터 저장 완료: {len(df_to_save)} 행")
            
        except Exception as e:
            logger.error(f"종목코드 {code} 데이터 저장 중 오류 발생: {str(e)}")
    
    def load_stock_data(self, code, start_date=None, end_date=None):
        """
        데이터베이스에서 주식 데이터 로드
        
        Parameters:
        -----------
        code : str
            종목 코드
        start_date : str
            시작 날짜 (YYYY-MM-DD)
        end_date : str
            종료 날짜 (YYYY-MM-DD)
            
        Returns:
        --------
        pandas.DataFrame
            로드된 주식 데이터
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 쿼리 작성
            query = "SELECT date, open, high, low, close, volume, change FROM stock_data WHERE code = ?"
            params = [code]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            # 데이터 로드
            df = pd.read_sql_query(query, conn, params=params)
            
            # 날짜 인덱스로 변환
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # 컬럼명 대문자로 변경
                df.columns = [col.capitalize() for col in df.columns]
            
            conn.close()
            
            logger.info(f"종목코드 {code} 데이터 로드 완료: {len(df)} 행")
            
            return df
            
        except Exception as e:
            logger.error(f"종목코드 {code} 데이터 로드 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def get_last_update_time(self, code):
        """
        종목의 마지막 업데이트 시간 조회
        
        Parameters:
        -----------
        code : str
            종목 코드
            
        Returns:
        --------
        datetime or None
            마지막 업데이트 시간
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT timestamp FROM last_update WHERE code = ?",
                (code,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            else:
                return None
                
        except Exception as e:
            logger.error(f"종목코드 {code} 마지막 업데이트 시간 조회 중 오류 발생: {str(e)}")
            return None
    
    def delete_stock_data(self, code=None):
        """
        주식 데이터 삭제
        
        Parameters:
        -----------
        code : str or None
            종목 코드 (None이면 모든 데이터 삭제)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if code:
                cursor.execute("DELETE FROM stock_data WHERE code = ?", (code,))
                cursor.execute("DELETE FROM last_update WHERE code = ?", (code,))
                logger.info(f"종목코드 {code} 데이터 삭제 완료")
            else:
                cursor.execute("DELETE FROM stock_data")
                cursor.execute("DELETE FROM last_update")
                logger.info("모든 주식 데이터 삭제 완료")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"주식 데이터 삭제 중 오류 발생: {str(e)}")
    
    def get_available_codes(self):
        """
        데이터베이스에 저장된 종목 코드 목록 조회
        
        Returns:
        --------
        list
            종목 코드 목록
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT code FROM stock_data")
            codes = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return codes
            
        except Exception as e:
            logger.error(f"종목 코드 목록 조회 중 오류 발생: {str(e)}")
            return []
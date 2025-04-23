import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import streamlit as st
from config import KIS_API

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('kis_api')

class KoreaInvestmentAPI:
    """한국투자증권 API 클래스"""
    
    def __init__(self):
        """초기화"""
        self.app_key = KIS_API['mock_app_key'] if KIS_API['use_mock'] else KIS_API['app_key']
        self.app_secret = KIS_API['mock_app_secret'] if KIS_API['use_mock'] else KIS_API['app_secret']
        self.acc_no = KIS_API['mock_acc_no'] if KIS_API['use_mock'] else KIS_API['acc_no']
        self.base_url = KIS_API['mock_url'] if KIS_API['use_mock'] else KIS_API['base_url']
        self.token_path = KIS_API['token_path']
        self.access_token = None
        self.token_expired_at = None
        
        # 클라우드 환경에서는 세션 상태 사용, 로컬에서는 파일 사용
        self.use_session_state = os.getenv('USE_SESSION_STATE', 'False').lower() == 'true'
        
        # 토큰 디렉토리 생성 (파일 저장 모드일 경우)
        if not self.use_session_state:
            token_dir = os.path.dirname(self.token_path)
            if token_dir and not os.path.exists(token_dir):
                os.makedirs(token_dir)
        
        # 토큰 로드 또는 발급
        self._load_or_issue_token()
    
    def _load_or_issue_token(self):
        """토큰 로드 또는 발급"""
        try:
            token_loaded = False
            
            # 세션 상태에서 토큰 로드 (클라우드 환경)
            if self.use_session_state:
                if 'kis_token' in st.session_state and 'expired_at' in st.session_state:
                    expired_at = datetime.fromtimestamp(st.session_state.expired_at)
                    if expired_at > datetime.now():
                        self.access_token = st.session_state.kis_token
                        self.token_expired_at = expired_at
                        logger.info("토큰을 세션 상태에서 로드했습니다.")
                        token_loaded = True
            
            # 파일에서 토큰 로드 (로컬 환경)
            elif os.path.exists(self.token_path):
                with open(self.token_path, 'r') as f:
                    token_data = json.load(f)
                
                # 토큰 만료 시간 확인
                expired_at = datetime.fromtimestamp(token_data['expired_at'])
                if expired_at > datetime.now():
                    self.access_token = token_data['access_token']
                    self.token_expired_at = expired_at
                    logger.info("토큰을 파일에서 로드했습니다.")
                    token_loaded = True
            
            # 토큰이 로드되지 않았거나 만료된 경우 새로 발급
            if not token_loaded:
                self._issue_token()
            
        except Exception as e:
            logger.error(f"토큰 로드 또는 발급 중 오류 발생: {str(e)}")
            raise
    
    def _issue_token(self):
        """토큰 발급"""
        url = f"{self.base_url}/oauth2/tokenP"
        
        headers = {
            "content-type": "application/json"
        }
        
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            # 토큰 만료 시간 설정 (발급 후 1일)
            self.token_expired_at = datetime.now() + timedelta(days=1)
            
            # 토큰 저장 (환경에 따라 세션 상태 또는 파일에 저장)
            if self.use_session_state:
                # Streamlit 세션 상태에 저장 (클라우드 환경)
                st.session_state.kis_token = self.access_token
                st.session_state.expired_at = self.token_expired_at.timestamp()
                logger.info("새로운 토큰을 세션 상태에 저장했습니다.")
            else:
                # 파일에 저장 (로컬 환경)
                with open(self.token_path, 'w') as f:
                    json.dump({
                        'access_token': self.access_token,
                        'expired_at': self.token_expired_at.timestamp()
                    }, f)
                logger.info("새로운 토큰을 파일에 저장했습니다.")
            
            logger.info("새로운 토큰을 발급받았습니다.")
            
        except Exception as e:
            logger.error(f"토큰 발급 중 오류 발생: {str(e)}")
            raise
    
    def _check_token(self):
        """토큰 유효성 확인 및 필요시 재발급"""
        if not self.access_token or not self.token_expired_at or self.token_expired_at <= datetime.now():
            self._issue_token()
    
    def get_stock_ohlcv(self, code, start_date=None, end_date=None, period=None):
        """주식 OHLCV 데이터 조회
        
        Args:
            code (str): 종목코드
            start_date (str, optional): 시작일자 (YYYYMMDD)
            end_date (str, optional): 종료일자 (YYYYMMDD)
            period (int, optional): 기간 (일)
        
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        self._check_token()
        
        # 날짜 설정
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if not start_date and period:
            start_date = (datetime.now() - timedelta(days=period)).strftime('%Y%m%d')
        elif not start_date:
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y%m%d')
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010400"  # 국내주식 일/주/월/년 시세
        }
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장구분: J(주식)
            "fid_input_iscd": code,  # 종목코드
            "fid_org_adj_prc": "1",  # 수정주가 여부: 1(수정주가)
            "fid_period_div_code": "D",  # 기간분류코드: D(일), W(주), M(월)
            "std_inpt_nb": "1",  # 표준입력개수
            "start_dt": start_date,  # 시작일자
            "end_dt": end_date  # 종료일자
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'output' not in data or 'output1' not in data:
                logger.warning(f"종목코드 {code}에 대한 데이터가 없습니다.")
                return pd.DataFrame()
            
            # 데이터 변환
            ohlcv_data = []
            for item in data['output1']:
                ohlcv_data.append({
                    'Date': pd.to_datetime(item['stck_bsop_date'], format='%Y%m%d'),
                    'Open': float(item['stck_oprc']),
                    'High': float(item['stck_hgpr']),
                    'Low': float(item['stck_lwpr']),
                    'Close': float(item['stck_clpr']),
                    'Volume': int(item['acml_vol'])
                })
            
            df = pd.DataFrame(ohlcv_data)
            
            if not df.empty:
                # 인덱스 설정
                df.set_index('Date', inplace=True)
                
                # 날짜 오름차순 정렬
                df = df.sort_index()
                
                # Change 컬럼 추가
                df['Change'] = df['Close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"종목코드 {code} 데이터 조회 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_current_price(self, code):
        """주식 현재가 조회
        
        Args:
            code (str): 종목코드
        
        Returns:
            dict: 현재가 정보
        """
        self._check_token()
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010100"  # 국내주식 현재가 시세
        }
        
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장구분: J(주식)
            "fid_input_iscd": code  # 종목코드
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'output' not in data:
                logger.warning(f"종목코드 {code}에 대한 현재가 정보가 없습니다.")
                return None
            
            output = data['output']
            
            # 현재가 정보 변환
            price_info = {
                'Open': float(output['stck_oprc']),
                'High': float(output['stck_hgpr']),
                'Low': float(output['stck_lwpr']),
                'Close': float(output['stck_prpr']),
                'Volume': int(output['acml_vol']),
                'Change': float(output['prdy_ctrt']) / 100  # 전일대비 등락률 (%)
            }
            
            return price_info
            
        except Exception as e:
            logger.error(f"종목코드 {code} 현재가 조회 중 오류 발생: {str(e)}")
            return None

    def get_account_balance(self):
        """계좌 잔고 조회
        
        Returns:
            dict: 계좌 잔고 정보
        """
        self._check_token()
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "VTTC8434R" if KIS_API['use_mock'] else "TTTC8434R"  # 모의투자/실전투자 구분
        }
        
        params = {
            "CANO": self.acc_no[:8],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.acc_no[8:],  # 계좌상품코드
            "AFHR_FLPR_YN": "N",  # 시간외단일가여부
            "OFL_YN": "N",  # 오프라인여부
            "INQR_DVSN": "02",  # 조회구분 (01: 추정조회, 02: 일반조회)
            "UNPR_DVSN": "01",  # 단가구분 (01: 원화, 02: 외화)
            "FUND_STTL_ICLD_YN": "N",  # 펀드결제분포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액자동상환여부
            "PRCS_DVSN": "01",  # 처리구분 (00: 전체, 01: 개별)
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""  # 연속조회키
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'output2' not in data:
                logger.warning("계좌 잔고 정보가 없습니다.")
                return None
            
            # 계좌 요약 정보
            account_summary = data['output2'][0] if data['output2'] else {}
            
            # 보유 종목 정보
            holdings = []
            if 'output1' in data:
                for item in data['output1']:
                    holdings.append({
                        'code': item['pdno'],  # 종목코드
                        'name': item['prdt_name'],  # 종목명
                        'quantity': int(item['hldg_qty']),  # 보유수량
                        'avg_price': float(item['pchs_avg_pric']),  # 매입평균가격
                        'current_price': float(item['prpr']),  # 현재가
                        'profit_loss': float(item['evlu_pfls_amt']),  # 평가손익금액
                        'profit_loss_rate': float(item['evlu_pfls_rt']),  # 평가손익률
                        'value': float(item['evlu_amt'])  # 평가금액
                    })
            
            # 계좌 잔고 정보
            balance_info = {
                'total_balance': float(account_summary.get('tot_evlu_amt', 0)),  # 총평가금액
                'cash_balance': float(account_summary.get('dnca_tot_amt', 0)),  # 예수금총금액
                'stock_value': float(account_summary.get('scts_evlu_amt', 0)),  # 유가증권평가금액
                'profit_loss': float(account_summary.get('evlu_pfls_smtl_amt', 0)),  # 평가손익합계금액
                'profit_loss_rate': float(account_summary.get('evlu_pfls_rt', 0)),  # 평가손익률
                'holdings': holdings  # 보유종목 목록
            }
            
            return balance_info
            
        except Exception as e:
            logger.error(f"계좌 잔고 조회 중 오류 발생: {str(e)}")
            return None
    
    def place_order(self, code, order_type, quantity, price=0, order_condition="00"):
        """주문 실행
        
        Args:
            code (str): 종목코드
            order_type (str): 주문유형 (1: 매수, 2: 매도)
            quantity (int): 주문수량
            price (int, optional): 주문가격 (시장가 주문인 경우 0)
            order_condition (str, optional): 주문조건 (00: 지정가, 01: 시장가)
        
        Returns:
            dict: 주문 결과
        """
        self._check_token()
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        # 모의투자/실전투자 구분
        tr_id = ""
        if KIS_API['use_mock']:
            if order_type == "1":  # 매수
                tr_id = "VTTC0802U"
            else:  # 매도
                tr_id = "VTTC0801U"
        else:
            if order_type == "1":  # 매수
                tr_id = "TTTC0802U"
            else:  # 매도
                tr_id = "TTTC0801U"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
        
        data = {
            "CANO": self.acc_no[:8],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.acc_no[8:],  # 계좌상품코드
            "PDNO": code,  # 종목코드
            "ORD_DVSN": order_condition,  # 주문조건 (00: 지정가, 01: 시장가)
            "ORD_QTY": str(quantity),  # 주문수량
            "ORD_UNPR": str(price) if price > 0 else "0",  # 주문단가
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            
            # 주문 결과 확인
            if 'output' in result:
                order_result = {
                    'order_no': result['output'].get('ODNO', ''),  # 주문번호
                    'order_time': result['output'].get('ORD_TMD', ''),  # 주문시각
                    'message': result.get('msg1', '')  # 결과 메시지
                }
                
                logger.info(f"주문 실행 성공: {order_result}")
                return order_result
            else:
                logger.warning(f"주문 실행 실패: {result}")
                return None
            
        except Exception as e:
            logger.error(f"주문 실행 중 오류 발생: {str(e)}")
            return None
    
    def get_order_history(self, start_date=None, end_date=None):
        """주문 내역 조회
        
        Args:
            start_date (str, optional): 시작일자 (YYYYMMDD)
            end_date (str, optional): 종료일자 (YYYYMMDD)
        
        Returns:
            list: 주문 내역 목록
        """
        self._check_token()
        
        # 날짜 설정
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "VTTC8001R" if KIS_API['use_mock'] else "TTTC8001R"  # 모의투자/실전투자 구분
        }
        
        params = {
            "CANO": self.acc_no[:8],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.acc_no[8:],  # 계좌상품코드
            "INQR_STRT_DT": start_date,  # 조회시작일자
            "INQR_END_DT": end_date,  # 조회종료일자
            "SLL_BUY_DVSN_CD": "00",  # 매도매수구분코드 (00: 전체, 01: 매도, 02: 매수)
            "INQR_DVSN": "00",  # 조회구분 (00: 역순, 01: 정순)
            "PDNO": "",  # 종목코드
            "CCLD_DVSN": "00",  # 체결구분 (00: 전체, 01: 체결, 02: 미체결)
            "ORD_GNO_BRNO": "",  # 주문채번지점번호
            "ODNO": "",  # 주문번호
            "INQR_DVSN_3": "",  # 조회구분3
            "INQR_DVSN_1": "",  # 조회구분1
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""  # 연속조회키
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'output1' not in data:
                logger.warning("주문 내역이 없습니다.")
                return []
            
            # 주문 내역 변환
            order_history = []
            for item in data['output1']:
                order_history.append({
                    'order_date': item.get('ORD_DT', ''),  # 주문일자
                    'order_time': item.get('ORD_TMD', ''),  # 주문시각
                    'code': item.get('PDNO', ''),  # 종목코드
                    'name': item.get('PRDT_NAME', ''),  # 종목명
                    'order_type': '매수' if item.get('SLL_BUY_DVSN_CD', '') == '02' else '매도',  # 매수/매도 구분
                    'quantity': int(item.get('ORD_QTY', '0')),  # 주문수량
                    'price': float(item.get('ORD_UNPR', '0')),  # 주문단가
                    'executed_quantity': int(item.get('CCLD_QTY', '0')),  # 체결수량
                    'executed_price': float(item.get('CCLD_UNPR', '0')),  # 체결단가
                    'order_status': item.get('CCLD_DVSN_NAME', ''),  # 체결구분명
                    'order_no': item.get('ODNO', '')  # 주문번호
                })
            
            return order_history
            
        except Exception as e:
            logger.error(f"주문 내역 조회 중 오류 발생: {str(e)}")
            return []

# 테스트 코드
if __name__ == "__main__":
    api = KoreaInvestmentAPI()
    
    # 삼성전자 데이터 조회
    df = api.get_stock_ohlcv('005930', period=30)
    print(df.tail())
    
    # 현재가 조회
    price = api.get_stock_current_price('005930')
    print(price)
    
    # 계좌 잔고 조회
    balance = api.get_account_balance()
    print("계좌 잔고:", balance)
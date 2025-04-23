import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from stock_data import StockData
from technical_indicators import TechnicalIndicators
from trading_signals import TradingSignals
from config import STOCK_CODES, UPDATE_INTERVAL, BACKTEST

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

class KoreanStockTradingBot:
    def __init__(self):
        logger.info("한국 주식 트레이딩 봇 초기화 중...")
        
        # 결과 저장 디렉토리 생성
        os.makedirs("results", exist_ok=True)
        
        # 모듈 초기화
        self.stock_data = StockData()
        self.technical_indicators = TechnicalIndicators()
        self.trading_signals = TradingSignals()
        
        # 현재 포지션 및 자산 정보
        # 각 종목별 포지션 정보 (수량, 평균 매수가, 손절매 가격, 익절매 가격)
        self.positions = {code: {'quantity': 0, 'avg_price': 0, 'stop_loss': 0, 'take_profit': 0} for code in STOCK_CODES}
        self.cash = 10000000  # 초기 현금 (1천만원)
        self.initial_capital = self.cash
        self.portfolio_value = self.cash
        self.transaction_history = []
        
        # 위험 관리 설정
        self.risk_management = BACKTEST['risk_management']
        
        logger.info("한국 주식 트레이딩 봇 초기화 완료")
    
    def update_data_and_signals(self):
        """데이터 업데이트 및 신호 생성"""
        # 데이터 업데이트
        self.stock_data.update_data()
        
        signals = {}
        for code in STOCK_CODES:
            if code in self.stock_data.data and not self.stock_data.data[code].empty:
                # 기술적 지표 계산
                df = self.technical_indicators.calculate_all(self.stock_data.data[code])
                
                # 매매 신호 생성
                df_with_signals = self.trading_signals.generate_signals(df)
                
                # 최신 신호 저장
                signals[code] = self.trading_signals.get_latest_signals(df_with_signals)
                
                # 데이터 업데이트
                self.stock_data.data[code] = df_with_signals
        
        return signals
    
    def execute_trades(self, signals):
        """매매 신호에 따라 거래 실행"""
        for code, signal in signals.items():
            if signal is None:
                continue
            
            # 현재 가격
            current_price = signal['price']
            
            # 매매 결정
            decision = signal['trading_decision']
            
            # 현재 포지션 정보
            position = self.positions[code]
            current_quantity = position['quantity']
            avg_price = position['avg_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            # 손절매/익절매 확인 (보유 중인 경우)
            if current_quantity > 0:
                # 손절매 조건 확인
                if stop_loss > 0 and current_price <= stop_loss:
                    # 손절매 실행
                    sell_amount = current_quantity * current_price
                    self.cash += sell_amount
                    
                    # 거래 기록
                    transaction = {
                        'timestamp': datetime.now(),
                        'code': code,
                        'action': 'STOP_LOSS',
                        'price': current_price,
                        'quantity': current_quantity,
                        'amount': sell_amount
                    }
                    self.transaction_history.append(transaction)
                    
                    # 포지션 초기화
                    self.positions[code] = {'quantity': 0, 'avg_price': 0, 'stop_loss': 0, 'take_profit': 0}
                    
                    logger.info(f"손절매 실행: {code}, 가격: {current_price}, 수량: {current_quantity}, 금액: {sell_amount}")
                    continue  # 다음 종목으로
                
                # 익절매 조건 확인
                if take_profit > 0 and current_price >= take_profit:
                    # 익절매 실행
                    sell_amount = current_quantity * current_price
                    self.cash += sell_amount
                    
                    # 거래 기록
                    transaction = {
                        'timestamp': datetime.now(),
                        'code': code,
                        'action': 'TAKE_PROFIT',
                        'price': current_price,
                        'quantity': current_quantity,
                        'amount': sell_amount
                    }
                    self.transaction_history.append(transaction)
                    
                    # 포지션 초기화
                    self.positions[code] = {'quantity': 0, 'avg_price': 0, 'stop_loss': 0, 'take_profit': 0}
                    
                    logger.info(f"익절매 실행: {code}, 가격: {current_price}, 수량: {current_quantity}, 금액: {sell_amount}")
                    continue  # 다음 종목으로
            
            # 매매 신호에 따른 거래 실행
            if decision == 1:  # 매수 신호
                # 최대 포지션 비율 확인
                max_position_pct = self.risk_management['max_position_pct']
                max_position_amount = self.initial_capital * (max_position_pct / 100)
                
                # 현재 포지션 가치
                current_position_value = current_quantity * current_price
                
                # 추가 투자 가능 금액 계산
                available_amount = max(0, max_position_amount - current_position_value)
                
                # 현재 현금의 10%와 가용 금액 중 작은 값으로 투자
                investment_amount = min(self.cash * 0.1, available_amount)
                
                if investment_amount > 0:
                    # 매수 수량 계산 (소수점 이하 버림)
                    quantity = int(investment_amount / current_price)
                    
                    if quantity > 0:
                        # 실제 투자 금액
                        actual_investment = quantity * current_price
                        
                        # 평균 매수가 계산
                        total_quantity = current_quantity + quantity
                        total_investment = (current_quantity * avg_price) + actual_investment
                        new_avg_price = total_investment / total_quantity if total_quantity > 0 else 0
                        
                        # 손절매/익절매 가격 설정
                        stop_loss_pct = self.risk_management['stop_loss_pct']
                        take_profit_pct = self.risk_management['take_profit_pct']
                        
                        new_stop_loss = new_avg_price * (1 - stop_loss_pct / 100)
                        new_take_profit = new_avg_price * (1 + take_profit_pct / 100)
                        
                        # 거래 실행
                        self.positions[code] = {
                            'quantity': total_quantity,
                            'avg_price': new_avg_price,
                            'stop_loss': new_stop_loss,
                            'take_profit': new_take_profit
                        }
                        self.cash -= actual_investment
                        
                        # 거래 기록
                        transaction = {
                            'timestamp': datetime.now(),
                            'code': code,
                            'action': 'BUY',
                            'price': current_price,
                            'quantity': quantity,
                            'amount': actual_investment,
                            'stop_loss': new_stop_loss,
                            'take_profit': new_take_profit
                        }
                        self.transaction_history.append(transaction)
                        
                        logger.info(f"매수 실행: {code}, 가격: {current_price}, 수량: {quantity}, 금액: {actual_investment}, "
                                   f"손절가: {new_stop_loss:.2f}, 익절가: {new_take_profit:.2f}")
            
            elif decision == -1:  # 매도 신호
                # 보유 수량
                quantity = current_quantity
                
                if quantity > 0:
                    # 매도 금액
                    sell_amount = quantity * current_price
                    
                    # 거래 실행
                    self.positions[code] = {'quantity': 0, 'avg_price': 0, 'stop_loss': 0, 'take_profit': 0}
                    self.cash += sell_amount
                    
                    # 거래 기록
                    transaction = {
                        'timestamp': datetime.now(),
                        'code': code,
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': quantity,
                        'amount': sell_amount
                    }
                    self.transaction_history.append(transaction)
                    
                    logger.info(f"매도 실행: {code}, 가격: {current_price}, 수량: {quantity}, 금액: {sell_amount}")
    
    def update_portfolio_value(self):
        """포트폴리오 가치 업데이트"""
        stock_value = 0
        
        for code, position in self.positions.items():
            quantity = position['quantity']
            if quantity > 0:
                latest_price = self.stock_data.get_latest_price(code)
                if latest_price is not None:
                    stock_value += quantity * latest_price['Close']
        
        self.portfolio_value = self.cash + stock_value
        return self.portfolio_value
    
    def print_status(self):
        """현재 상태 출력"""
        logger.info("=" * 50)
        logger.info(f"현재 시간: {datetime.now()}")
        logger.info(f"현금: {self.cash:,.0f}원")
        
        stock_value = 0
        logger.info("보유 종목:")
        for code, position in self.positions.items():
            quantity = position['quantity']
            avg_price = position['avg_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if quantity > 0:
                latest_price = self.stock_data.get_latest_price(code)
                if latest_price is not None:
                    current_price = latest_price['Close']
                    value = quantity * current_price
                    profit_loss = (current_price - avg_price) * quantity
                    profit_loss_pct = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
                    
                    stock_value += value
                    logger.info(f"  - {code}: {quantity}주, 평균단가: {avg_price:,.0f}원, 현재가: {current_price:,.0f}원, "
                               f"평가금액: {value:,.0f}원, 손익: {profit_loss:,.0f}원 ({profit_loss_pct:.2f}%), "
                               f"손절가: {stop_loss:,.0f}원, 익절가: {take_profit:,.0f}원")
        
        total_value = self.cash + stock_value
        profit_loss = total_value - self.initial_capital
        profit_loss_pct = (profit_loss / self.initial_capital) * 100
        
        logger.info(f"주식 평가금액: {stock_value:,.0f}원")
        logger.info(f"총 자산: {total_value:,.0f}원")
        logger.info(f"손익: {profit_loss:,.0f}원 ({profit_loss_pct:.2f}%)")
        logger.info("=" * 50)
    
    def plot_portfolio_performance(self, portfolio_history):
        """포트폴리오 성과 그래프 생성"""
        if not portfolio_history:
            return
        
        # 데이터 준비
        dates = [entry['timestamp'] for entry in portfolio_history]
        values = [entry['value'] for entry in portfolio_history]
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Value (KRW)')
        plt.legend()
        plt.grid(True)
        
        # 파일로 저장
        plt.savefig('results/portfolio_performance.png')
        logger.info("포트폴리오 성과 그래프가 저장되었습니다: results/portfolio_performance.png")
    
    def run_simulation(self, duration_seconds=300):
        """트레이딩 봇 시뮬레이션 실행"""
        logger.info(f"트레이딩 봇 시뮬레이션 시작 (실행 시간: {duration_seconds}초)")
        
        start_time = time.time()
        portfolio_history = []
        
        try:
            while time.time() - start_time < duration_seconds:
                # 실시간 데이터 시뮬레이션
                self.stock_data.simulate_realtime_data()
                
                # 신호 업데이트 및 거래 실행
                signals = self.update_data_and_signals()
                self.execute_trades(signals)
                
                # 포트폴리오 가치 업데이트
                portfolio_value = self.update_portfolio_value()
                
                # 포트폴리오 히스토리 기록
                portfolio_history.append({
                    'timestamp': datetime.now(),
                    'value': portfolio_value
                })
                
                # 현재 상태 출력
                self.print_status()
                
                # 대기
                time.sleep(UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("사용자에 의해 시뮬레이션이 중단되었습니다.")
        except Exception as e:
            logger.error(f"시뮬레이션 중 오류 발생: {str(e)}")
        finally:
            # 최종 결과 출력
            self.print_status()
            
            # 포트폴리오 성과 그래프 생성
            self.plot_portfolio_performance(portfolio_history)
            
            # 거래 내역 저장
            self.save_transaction_history()
            
            logger.info("트레이딩 봇 시뮬레이션 종료")
    
    def save_transaction_history(self):
        """거래 내역 저장"""
        if not self.transaction_history:
            logger.info("저장할 거래 내역이 없습니다.")
            return
        
        # 데이터프레임 생성
        df = pd.DataFrame(self.transaction_history)
        
        # CSV 파일로 저장
        df.to_csv('results/transaction_history.csv', index=False)
        logger.info("거래 내역이 저장되었습니다: results/transaction_history.csv")
    
    def run_backtest(self):
        """과거 데이터로 백테스팅 실행"""
        logger.info("백테스팅 시작")
        
        backtest_results = {}
        
        for code in STOCK_CODES:
            if code in self.stock_data.data and not self.stock_data.data[code].empty:
                logger.info(f"종목코드 {code} 백테스팅 중...")
                
                # 기술적 지표 계산
                df = self.technical_indicators.calculate_all(self.stock_data.data[code])
                
                # 매매 신호 생성
                df_with_signals = self.trading_signals.generate_signals(df)
                
                # 백테스팅 수행
                result = self.trading_signals.backtest(df_with_signals, BACKTEST['initial_capital'])
                
                if result:
                    backtest_results[code] = result
                    
                    # 결과 출력
                    logger.info(f"종목코드 {code} 백테스팅 결과:")
                    logger.info(f"  - 초기 자본: {result['initial_capital']:,.0f}원")
                    logger.info(f"  - 최종 자본: {result['final_capital']:,.0f}원")
                    logger.info(f"  - 수익률: {result['total_return_pct']:.2f}%")
                    logger.info(f"  - 거래 횟수: {result['num_trades']}")
                    logger.info(f"  - 최대 낙폭: {result['max_drawdown_pct']:.2f}%")
                    
                    # 백테스팅 결과 그래프 생성
                    self._plot_backtest_result(code, result)
        
        logger.info("백테스팅 완료")
        return backtest_results
    
    def _plot_backtest_result(self, code, result):
        """백테스팅 결과 그래프 생성"""
        if 'backtest_data' not in result:
            return
        
        df = result['backtest_data']
        
        # 그래프 생성
        plt.figure(figsize=(12, 8))
        
        # 수익률 그래프
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Cumulative_Returns'], label='Buy & Hold')
        plt.plot(df.index, df['Strategy_Cumulative_Returns'], label='Strategy')
        plt.title(f'{code} Backtest Result - Cumulative Returns')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        # 자본금 그래프
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['Capital'])
        plt.title(f'{code} Backtest Result - Capital')
        plt.ylabel('Capital (KRW)')
        plt.grid(True)
        
        # 파일로 저장
        plt.tight_layout()
        plt.savefig(f'results/backtest_{code}.png')
        logger.info(f"백테스팅 결과 그래프가 저장되었습니다: results/backtest_{code}.png")

def main():
    """메인 함수"""
    logger.info("한국 주식 트레이딩 봇 시작")
    
    # 트레이딩 봇 인스턴스 생성
    bot = KoreanStockTradingBot()
    
    # 백테스팅 실행
    bot.run_backtest()
    
    # 실시간 시뮬레이션 실행 (5분)
    bot.run_simulation(duration_seconds=300)
    
    logger.info("한국 주식 트레이딩 봇 종료")

if __name__ == "__main__":
    main()
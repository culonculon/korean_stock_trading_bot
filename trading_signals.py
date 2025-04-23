import pandas as pd
import numpy as np
import logging
from config import TRADING_SIGNALS
from utils import handle_missing_values

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_signals')

class TradingSignals:
    def __init__(self):
        self.config = TRADING_SIGNALS
        logger.info("매매 신호 생성기 초기화 완료")
    
    def generate_signals(self, df):
        """모든 매매 신호 생성"""
        if df is None or df.empty:
            logger.warning("데이터가 비어있어 매매 신호를 생성할 수 없습니다.")
            return df
        
        # 데이터 복사
        result = df.copy()
        
        # 결측치 확인
        missing_values = result.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"매매 신호 생성 전 결측치 발견: {missing_values}")
            
            # 가격 데이터 결측치 처리
            price_columns = ['Close']
            result = handle_missing_values(result, method='ffill_bfill', columns=price_columns)
            
            # 신호 데이터 결측치 처리 (0으로 채우기)
            signal_columns = ['SMA_Signal', 'RSI_Signal', 'MACD_Signal', 'BB_Signal']
            for col in signal_columns:
                if col in result.columns and isinstance(result[col], pd.Series):
                    null_check = result[col].isnull()
                    if null_check.any():
                        result[col] = result[col].fillna(0)
                elif col in result.columns:
                    logger.warning(f"{col}은(는) Series가 아닙니다.")
                else:
                    logger.warning(f"{col}은(는) 존재하지 않습니다.")
        
        # 최종 매매 신호 컬럼 추가
        result['Final_Signal'] = 0
        
        # 각 전략별 신호 생성 및 통합
        if self.config['sma_crossover']:
            self._apply_sma_crossover_strategy(result)
        
        if self.config['rsi_bounds']:
            self._apply_rsi_bounds_strategy(result)
        
        if self.config['macd_crossover']:
            self._apply_macd_crossover_strategy(result)
        
        if self.config['bollinger_breakout']:
            self._apply_bollinger_breakout_strategy(result)
            
        if self.config['stochastic_crossover']:
            self._apply_stochastic_crossover_strategy(result)
        
        # 신호 강도 계산 (여러 지표가 동시에 같은 신호를 보내면 강도가 높아짐)
        result['Signal_Strength'] = result['Final_Signal'].abs()
        
        # 매매 결정 (1: 매수, -1: 매도, 0: 홀드)
        result['Trading_Decision'] = np.sign(result['Final_Signal'])
        
        logger.info("매매 신호 생성 완료")
        return result
    
    def _apply_sma_crossover_strategy(self, df):
        """SMA 크로스오버 전략 적용"""
        try:
            if 'SMA_Signal' in df.columns:
                # SMA 신호 반영
                df['Final_Signal'] += df['SMA_Signal']
                logger.info("SMA 크로스오버 전략 적용 완료")
            else:
                logger.warning("SMA 신호가 없어 SMA 크로스오버 전략을 적용할 수 없습니다.")
        except Exception as e:
            logger.error(f"SMA 크로스오버 전략 적용 중 오류 발생: {str(e)}")
    
    def _apply_rsi_bounds_strategy(self, df):
        """RSI 경계값 전략 적용"""
        try:
            if 'RSI_Signal' in df.columns:
                # RSI 신호 반영
                df['Final_Signal'] += df['RSI_Signal']
                logger.info("RSI 경계값 전략 적용 완료")
            else:
                logger.warning("RSI 신호가 없어 RSI 경계값 전략을 적용할 수 없습니다.")
        except Exception as e:
            logger.error(f"RSI 경계값 전략 적용 중 오류 발생: {str(e)}")
    
    def _apply_macd_crossover_strategy(self, df):
        """MACD 크로스오버 전략 적용"""
        try:
            if 'MACD_Signal' in df.columns:
                # MACD 신호 반영
                df['Final_Signal'] += df['MACD_Signal']
                logger.info("MACD 크로스오버 전략 적용 완료")
            else:
                logger.warning("MACD 신호가 없어 MACD 크로스오버 전략을 적용할 수 없습니다.")
        except Exception as e:
            logger.error(f"MACD 크로스오버 전략 적용 중 오류 발생: {str(e)}")
    
    def _apply_bollinger_breakout_strategy(self, df):
        """볼린저 밴드 돌파 전략 적용"""
        try:
            if 'BB_Signal' in df.columns:
                # 볼린저 밴드 신호 반영
                df['Final_Signal'] += df['BB_Signal']
                logger.info("볼린저 밴드 돌파 전략 적용 완료")
            else:
                logger.warning("볼린저 밴드 신호가 없어 볼린저 밴드 돌파 전략을 적용할 수 없습니다.")
        except Exception as e:
            logger.error(f"볼린저 밴드 돌파 전략 적용 중 오류 발생: {str(e)}")
    
    def _apply_stochastic_crossover_strategy(self, df):
        """스토캐스틱 크로스오버 전략 적용"""
        try:
            if 'Stochastic_Signal' in df.columns:
                # 스토캐스틱 신호 반영
                df['Final_Signal'] += df['Stochastic_Signal']
                logger.info("스토캐스틱 크로스오버 전략 적용 완료")
            else:
                logger.warning("스토캐스틱 신호가 없어 스토캐스틱 크로스오버 전략을 적용할 수 없습니다.")
        except Exception as e:
            logger.error(f"스토캐스틱 크로스오버 전략 적용 중 오류 발생: {str(e)}")
    
    def get_latest_signals(self, df):
        """최신 매매 신호 반환"""
        if df is None or df.empty:
            return None
        
        # 최신 데이터의 매매 신호 반환
        latest = df.iloc[-1]
        
        # 다중 인덱스 컬럼 처리
        def get_value(series, column_name):
            # 시리즈가 멀티인덱스를 사용하는 경우
            if isinstance(series.index, pd.MultiIndex):
                # 해당 컬럼 이름으로 시작하는 첫 번째 인덱스 찾기
                matching_indices = [idx for idx in series.index if idx[0] == column_name]
                if matching_indices:
                    return series[matching_indices[0]]
                return 0
            # 일반 인덱스를 사용하는 경우
            else:
                return series.get(column_name, 0)
        
        # Series 객체를 단일 값으로 변환
        signal_info = {
            'timestamp': latest.name,
            'price': float(get_value(latest, 'Close')),
            'trading_decision': int(get_value(latest, 'Trading_Decision')),
            'signal_strength': int(get_value(latest, 'Signal_Strength')),
            'signals': {
                'sma': int(get_value(latest, 'SMA_Signal')),
                'rsi': int(get_value(latest, 'RSI_Signal')),
                'macd': int(get_value(latest, 'MACD_Signal')),
                'bollinger': int(get_value(latest, 'BB_Signal')),
                'stochastic': int(get_value(latest, 'Stochastic_Signal'))
            }
        }
        
        return signal_info
    
    def backtest(self, df, initial_capital=10000000):
        """간단한 백테스팅 수행"""
        if df is None or df.empty or 'Trading_Decision' not in df.columns:
            logger.warning("백테스팅을 위한 데이터가 부족합니다.")
            return None
        
        # 결측치 확인 및 처리
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"백테스팅 전 결측치 발견: {missing_values}")
            
            # 백테스팅에 필요한 컬럼만 추출하여 결측치 처리
            df_clean = df.copy()
            
            # 'Close' 컬럼 결측치 처리 - ffill_bfill 방식 사용
            df_clean = handle_missing_values(df_clean, method='ffill_bfill', columns=['Close'])
            
            # 'Trading_Decision' 컬럼 결측치 처리 - 0으로 채우기
            if 'Trading_Decision' in df_clean.columns:
                df_clean = handle_missing_values(df_clean, method='zero', columns=['Trading_Decision'])
            else:
                logger.warning("'Trading_Decision' 컬럼이 없습니다.")
            
            # 여전히 결측치가 있는 필수 컬럼 확인
            required_cols = ['Close', 'Trading_Decision']
            
            # 다중 인덱스 처리를 위한 컬럼 찾기
            if isinstance(df_clean.columns, pd.MultiIndex):
                existing_cols = []
                for base_col in required_cols:
                    matching_cols = [col for col in df_clean.columns if col[0] == base_col]
                    existing_cols.extend(matching_cols)
            else:
                existing_cols = [col for col in required_cols if col in df_clean.columns]
            
            # 필수 컬럼에 결측치가 있는 행 제거
            if existing_cols:
                missing_after = df_clean[existing_cols].isnull().sum()
                if missing_after.sum() > 0:
                    logger.warning(f"결측치 처리 후에도 남아있는 결측치: {missing_after}")
                    df_clean = df_clean.dropna(subset=existing_cols)
                    
                    # 결측치가 너무 많이 제거되었는지 확인
                    if len(df_clean) < len(df) * 0.7:  # 30% 이상 제거되면 경고
                        logger.warning(f"결측치 제거로 인해 데이터의 {100 - len(df_clean)/len(df)*100:.1f}%가 손실되었습니다.")
            else:
                logger.warning(f"필수 컬럼을 찾을 수 없습니다: {required_cols}")

            if df_clean.empty:
                logger.warning("결측치 처리 후 데이터가 비어있어 백테스팅을 수행할 수 없습니다.")
                return None
            
            df = df_clean
        
        # 다중 인덱스 컬럼 처리를 위한 헬퍼 함수들
        def find_column(dataframe, name):
            """주어진 이름의 컬럼을 찾습니다 (다중 인덱스 지원)"""
            if isinstance(dataframe.columns, pd.MultiIndex):
                cols = [col for col in dataframe.columns if col[0] == name]
                return cols[0] if cols else None
            else:
                return name if name in dataframe.columns else None
                
        def add_column(dataframe, name, values):
            """데이터프레임에 새 컬럼을 추가합니다 (다중 인덱스 지원)"""
            if isinstance(dataframe.columns, pd.MultiIndex):
                # 기존 컬럼의 다중 인덱스 구조를 유지
                # (일반적으로 첫 번째 레벨은 컬럼 이름, 두 번째 레벨은 종목 코드)
                sample_col = dataframe.columns[0]
                if len(sample_col) > 1:
                    new_col = (name, sample_col[1])
                    dataframe[new_col] = values
                    return new_col
                else:
                    dataframe[name] = values
                    return name
            else:
                dataframe[name] = values
                return name
        
        # 백테스팅을 위한 데이터프레임 복사
        backtest_results = df.copy(deep=True)
        
        # 필요한 컬럼 찾기
        close_col = find_column(backtest_results, 'Close')
        trading_decision_col = find_column(backtest_results, 'Trading_Decision')
        
        if close_col is None or trading_decision_col is None:
            logger.warning("필요한 컬럼을 찾을 수 없습니다.")
            return None
        
        # 포지션 컬럼 추가 (1: 매수 포지션, 0: 현금 보유)
        position_values = pd.Series(0, index=backtest_results.index)
        position_col = add_column(backtest_results, 'Position', position_values)
        
        # 초기 포지션 설정 (첫 번째 행은 항상 0)
        if len(backtest_results) > 1:
            # 매매 신호에 따라 포지션 결정
            # 매수 신호(1)이 오면 포지션 진입, 매도 신호(-1)이 오면 포지션 청산
            for i in range(1, len(backtest_results)):
                prev_decision = backtest_results.iloc[i - 1][trading_decision_col]
                
                position_col_idx = backtest_results.columns.get_loc(position_col)
                
                if prev_decision == 1:  # 매수 신호
                    backtest_results.iloc[i, position_col_idx] = 1
                elif prev_decision == -1:  # 매도 신호
                    backtest_results.iloc[i, position_col_idx] = 0
                else:  # 홀드
                    backtest_results.iloc[i, position_col_idx] = backtest_results.iloc[i - 1][position_col]

        # === 수익률 계산 ===
        returns = backtest_results[close_col].pct_change(fill_method=None).fillna(0)
        returns_col = add_column(backtest_results, 'Returns', returns)

        # === 전략 수익률 ===
        strategy_returns = backtest_results[position_col] * backtest_results[returns_col]
        strategy_returns_col = add_column(backtest_results, 'Strategy_Returns', strategy_returns)

        # === 누적 수익률 ===
        cumulative_returns = (1 + backtest_results[returns_col]).cumprod()
        cumulative_returns_col = add_column(backtest_results, 'Cumulative_Returns', cumulative_returns)
        
        strategy_cumulative_returns = (1 + backtest_results[strategy_returns_col]).cumprod()
        strategy_cumulative_returns_col = add_column(backtest_results, 'Strategy_Cumulative_Returns', strategy_cumulative_returns)

        # === 자본금 계산 ===
        capital = initial_capital * backtest_results[strategy_cumulative_returns_col]
        capital_col = add_column(backtest_results, 'Capital', capital)

        # === 거래 횟수 계산 ===
        position_changes = backtest_results[position_col].diff().fillna(0)
        num_trades = (position_changes == 1).sum()

        # === MDD 계산 ===      
        peak = backtest_results[strategy_cumulative_returns_col].cummax()
        peak_col = add_column(backtest_results, 'Peak', peak)

        drawdown = (backtest_results[strategy_cumulative_returns_col] / backtest_results[peak_col]) - 1
        drawdown_col = add_column(backtest_results, 'Drawdown', drawdown)

        # max_drawdown을 단일 값으로 변환 
        max_drawdown = backtest_results['Drawdown'].min()
        if isinstance(max_drawdown, pd.Series):  # max_drawdown이 Series인지 확인
            max_drawdown = max_drawdown.min()  # 단일 값으로 변환

        if np.isnan(max_drawdown):  # NaN인지 확인
            max_drawdown = 0


        # === 최종 결과 ===
        if len(backtest_results) > 0:
            final_capital = backtest_results[capital_col].iloc[-1]
            if np.isnan(final_capital):
                logger.warning("최종 자본금이 NaN입니다. 초기값으로 대체합니다.")
                final_capital = initial_capital
                total_return = 0
            else:
                total_return = (final_capital / initial_capital - 1) * 100
        else:
            final_capital = initial_capital
            total_return = 0

        # === 추가 성능 지표 계산 ===
        # 일별 수익률
        daily_returns = backtest_results[strategy_returns_col]
        
        # 연간화된 수익률 (252 거래일 기준)
        annual_return = ((1 + total_return / 100) ** (252 / len(daily_returns)) - 1) * 100 if len(daily_returns) > 0 else 0
        
        # 변동성 (연간화)
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        
        # 샤프 비율 (무위험 수익률 2% 가정)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return / 100 - risk_free_rate) / (volatility / 100) if volatility > 0 else 0
        
        # 승률 계산
        winning_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) * 100 if (winning_days + losing_days) > 0 else 0
        
        # 평균 수익/손실 비율
        avg_profit = daily_returns[daily_returns > 0].mean() * 100 if len(daily_returns[daily_returns > 0]) > 0 else 0
        avg_loss = daily_returns[daily_returns < 0].mean() * 100 if len(daily_returns[daily_returns < 0]) > 0 else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # === 최종 반환 ===
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate,
            'avg_profit_pct': avg_profit,
            'avg_loss_pct': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'backtest_data': backtest_results
        }

        logger.info(f"백테스팅 완료: 수익률 {total_return:.2f}%, 연간 수익률 {annual_return:.2f}%, "
                   f"샤프 비율 {sharpe_ratio:.2f}, 승률 {win_rate:.2f}%, 최대 낙폭 {max_drawdown*100:.2f}%")
        return results


# 테스트 코드
if __name__ == "__main__":
    import yfinance as yf
    from pykrx import stock
    from datetime import datetime, timedelta
    from technical_indicators import TechnicalIndicators
    
    # 테스트 데이터 가져오기
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    # yfinance로 데이터 가져오기
    code = '005930'
    yf_code = f"{code}.KS"
    df = yf.download(
        yf_code,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )
    
    # yfinance로 데이터를 가져오지 못한 경우 pykrx 사용
    if df.empty:
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        df = stock.get_market_ohlcv_by_date(start_date_str, end_date_str, code)
        
        # 컬럼명 변경
        if not df.empty:
            df = df.rename(columns={
                '시가': 'Open',
                '고가': 'High',
                '저가': 'Low',
                '종가': 'Close',
                '거래량': 'Volume'
            })
            
            # Change 컬럼 추가
            df['Change'] = df['Close'].pct_change()
    
    # 기술적 지표 계산
    ti = TechnicalIndicators()
    df_with_indicators = ti.calculate_all(df)
    
    # 매매 신호 생성
    ts = TradingSignals()
    result = ts.generate_signals(df_with_indicators)
    
    # 최신 신호 출력
    latest_signal = ts.get_latest_signals(result)
    print("최신 매매 신호:", latest_signal)
    
    # 백테스팅 수행
    backtest_result = ts.backtest(result)
    print(f"백테스팅 결과: 수익률 {backtest_result['total_return_pct']:.2f}%")
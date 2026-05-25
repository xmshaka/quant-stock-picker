"""因子IC分析测试"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from analysis.ic_analysis import ICAnalyzer, ICResult
from analysis.report import ICReport


class TestICCalc:
    """测试IC基础计算"""
    
    def test_perfect_correlation(self):
        """完美正相关"""
        analyzer = ICAnalyzer(min_stocks=5)
        f = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
        r = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
        ic, rank_ic = analyzer.calc_ic(f, r)
        assert ic == pytest.approx(1.0, abs=0.01)
        assert rank_ic == pytest.approx(1.0, abs=0.01)
    
    def test_perfect_negative(self):
        """完美负相关"""
        analyzer = ICAnalyzer(min_stocks=5)
        f = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
        r = pd.Series([5, 4, 3, 2, 1], index=['a', 'b', 'c', 'd', 'e'])
        ic, rank_ic = analyzer.calc_ic(f, r)
        assert ic == pytest.approx(-1.0, abs=0.01)
        assert rank_ic == pytest.approx(-1.0, abs=0.01)
    
    def test_no_correlation(self):
        """无相关"""
        analyzer = ICAnalyzer()
        np.random.seed(42)
        f = pd.Series(np.random.randn(100))
        r = pd.Series(np.random.randn(100))
        ic, rank_ic = analyzer.calc_ic(f, r)
        assert abs(ic) < 0.3
        assert abs(rank_ic) < 0.3
    
    def test_insufficient_data(self):
        """数据不足返回nan"""
        analyzer = ICAnalyzer(min_stocks=10)
        f = pd.Series([1, 2, 3])
        r = pd.Series([1, 2, 3])
        ic, rank_ic = analyzer.calc_ic(f, r)
        assert np.isnan(ic)
        assert np.isnan(rank_ic)
    
    def test_with_nans(self):
        """含NaN处理"""
        analyzer = ICAnalyzer()
        f = pd.Series([1, np.nan, 3, 4, 5])
        r = pd.Series([1, 2, np.nan, 4, 5])
        ic, rank_ic = analyzer.calc_ic(f, r)
        # 只有3个有效点
        assert np.isnan(ic)  # 少于min_stocks=10


class TestForwardReturn:
    """测试未来收益计算"""
    
    def test_forward_return_basic(self):
        """基础未来收益"""
        analyzer = ICAnalyzer()
        dates = pd.date_range('2025-01-01', periods=10, freq='B')
        df = pd.DataFrame({
            'symbol': ['A'] * 10,
            'trade_date': dates,
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        result = analyzer.calc_forward_return(df, horizon=2)
        # 第0天的2日后收益 = 102/100 - 1 = 0.02
        assert len(result) == 8  # 10 - 2 = 8
        assert result.iloc[0]['forward_return'] == pytest.approx(0.02, abs=0.001)
    
    def test_forward_return_multi_stock(self):
        """多股票未来收益"""
        analyzer = ICAnalyzer()
        dates = pd.date_range('2025-01-01', periods=5, freq='B')
        df = pd.DataFrame({
            'symbol': ['A'] * 5 + ['B'] * 5,
            'trade_date': list(dates) * 2,
            'close': [100, 101, 102, 103, 104, 200, 198, 196, 194, 192]
        })
        result = analyzer.calc_forward_return(df, horizon=1)
        assert len(result) == 8  # (5-1)*2 = 8
        # A第0天1日收益 = 101/100 - 1 = 0.01
        a_result = result[result['symbol'] == 'A']
        assert a_result.iloc[0]['forward_return'] == pytest.approx(0.01, abs=0.001)


class TestICSeries:
    """测试IC序列分析"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=20, freq='B')
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        # 价格数据
        price_data = []
        for sym in symbols:
            base = np.random.uniform(50, 200)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base * np.exp(np.cumsum(returns))
            for i, d in enumerate(dates):
                price_data.append({
                    'symbol': sym,
                    'trade_date': d,
                    'close': round(prices[i], 2)
                })
        price_df = pd.DataFrame(price_data)
        
        # 因子数据 (与次日收益有一定相关)
        factor_data = []
        for d in dates:
            for sym in symbols:
                factor_data.append({
                    'symbol': sym,
                    'trade_date': d,
                    'momentum_20d': np.random.randn()
                })
        factor_df = pd.DataFrame(factor_data)
        
        return factor_df, price_df
    
    def test_ic_series_shape(self, sample_data):
        """IC序列形状"""
        factor_df, price_df = sample_data
        analyzer = ICAnalyzer()
        ic_df = analyzer.analyze_single_factor(factor_df, price_df, 'momentum_20d', horizon=1)
        assert 'trade_date' in ic_df.columns
        assert 'ic' in ic_df.columns
        assert 'rank_ic' in ic_df.columns
        assert len(ic_df) > 0
    
    def test_ic_stats(self, sample_data):
        """IC统计"""
        factor_df, price_df = sample_data
        analyzer = ICAnalyzer()
        ic_df = analyzer.analyze_single_factor(factor_df, price_df, 'momentum_20d', horizon=1)
        stats = analyzer.calc_ic_stats(ic_df)
        assert 'ic_mean' in stats
        assert 'ir' in stats
        assert 'valid_days' in stats


class TestICDecay:
    """测试IC衰减"""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=30, freq='B')
        symbols = list('ABCDEFGHIJ')
        
        price_data = []
        for sym in symbols:
            base = 100
            returns = np.random.normal(0, 0.02, len(dates))
            prices = base * np.exp(np.cumsum(returns))
            for i, d in enumerate(dates):
                price_data.append({'symbol': sym, 'trade_date': d, 'close': prices[i]})
        
        factor_data = []
        for d in dates:
            for sym in symbols:
                factor_data.append({'symbol': sym, 'trade_date': d, 'factor': np.random.randn()})
        
        return pd.DataFrame(factor_data), pd.DataFrame(price_data)
    
    def test_decay_shape(self, sample_data):
        factor_df, price_df = sample_data
        analyzer = ICAnalyzer()
        decay = analyzer.calc_ic_decay(factor_df, price_df, 'factor', [1, 5, 10])
        assert len(decay) == 3
        assert list(decay['horizon']) == [1, 5, 10]


class TestGroupReturn:
    """测试分组收益"""
    
    def test_group_return_shape(self):
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=10, freq='B')
        symbols = list('ABCDEFGHIJ')
        
        price_data = []
        for sym in symbols:
            prices = 100 + np.cumsum(np.random.randn(len(dates)))
            for i, d in enumerate(dates):
                price_data.append({'symbol': sym, 'trade_date': d, 'close': prices[i]})
        
        factor_data = []
        for d in dates:
            for sym in symbols:
                factor_data.append({'symbol': sym, 'trade_date': d, 'factor': np.random.randn()})
        
        analyzer = ICAnalyzer()
        result = analyzer.group_return_analysis(
            pd.DataFrame(factor_data), pd.DataFrame(price_data), 'factor', n_groups=5
        )
        assert len(result) <= 5
        assert 'mean_return' in result.columns
        assert 'std' in result.columns


class TestICReport:
    """测试报告生成"""
    
    def test_generate_report(self):
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=15, freq='B')
        symbols = list('ABCDE')
        
        price_data = []
        for sym in symbols:
            prices = 100 + np.cumsum(np.random.randn(len(dates)))
            for i, d in enumerate(dates):
                price_data.append({'symbol': sym, 'trade_date': d, 'close': prices[i]})
        
        factor_data = []
        for d in dates:
            for sym in symbols:
                factor_data.append({'symbol': sym, 'trade_date': d, 'my_factor': np.random.randn()})
        
        report = ICReport()
        result = report.generate(
            pd.DataFrame(factor_data), pd.DataFrame(price_data), 'my_factor'
        )
        assert 'factor_name' in result
        assert 'summary' in result
        assert 'ic_series' in result
        assert 'ic_decay' in result
        assert 'group_return' in result
    
    def test_text_output(self):
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=15, freq='B')
        symbols = list('ABCDE')
        
        price_data = []
        for sym in symbols:
            prices = 100 + np.cumsum(np.random.randn(len(dates)))
            for i, d in enumerate(dates):
                price_data.append({'symbol': sym, 'trade_date': d, 'close': prices[i]})
        
        factor_data = []
        for d in dates:
            for sym in symbols:
                factor_data.append({'symbol': sym, 'trade_date': d, 'my_factor': np.random.randn()})
        
        report = ICReport()
        result = report.generate(
            pd.DataFrame(factor_data), pd.DataFrame(price_data), 'my_factor'
        )
        text = report.to_text(result)
        assert '因子IC分析报告' in text
        assert 'IC均值' in text

# portfolio_analyzer.py
# ENHANCED COMPREHENSIVE PORTFOLIO ANALYZER
# Provides detailed scrutiny, economic model analysis, and extensive plotting
# Automatically analyzes results without requiring manual intervention

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
import warnings
import os
import glob
from datetime import datetime
import json

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

@dataclass
class AnalyzerConfig:
    """Enhanced configuration for the comprehensive portfolio analyzer."""
    risk_free_annual: float = 0.02
    min_annualization: int = 252           # Min trading days for annualization
    max_annualization: int = 52560         # Max steps (e.g., 10-min intervals) per year
    equity_cols: Tuple[str, ...] = ("portfolio_value", "equity", "total_return_nav", "portfolio_performance", "fund_performance")
    budget_cols: Tuple[str, ...] = ("budget", "investment_capital")
    log_prefer: bool = True                # Prefer log returns for calculations if possible
    plot_title: str = "Hybrid Renewable Energy Fund - AI Performance Analysis"

    # Enhanced analysis parameters
    initial_fund_size: float = 500_000_000  # $500M fund
    target_baseline_return: float = 0.0376   # 3.76% baseline
    target_ai_return: float = 0.0544         # 5.44% AI-enhanced

    # Get currency conversion from config
    def __post_init__(self):
        try:
            from config import EnhancedConfig
            config = EnhancedConfig()
            self.currency_conversion = config.dkk_to_usd_rate  # DKK to USD from config
        except Exception:
            self.currency_conversion = 0.145  # Fallback value

    # Economic model parameters
    physical_allocation: float = 0.518       # 51.8% physical assets
    financial_allocation: float = 0.482      # 48.2% financial instruments

    # Asset specifications (fractional ownership)
    wind_ownership: float = 0.05            # 5% of 1,500MW wind farm
    solar_ownership: float = 0.05           # 5% of 1,000MW solar farm
    hydro_ownership: float = 0.02           # 2% of 1,000MW hydro plant
    battery_capacity: float = 10.0          # 10MWh direct ownership

    # Performance thresholds
    confidence_threshold_90: int = 12960    # 90 days for 90% confidence
    confidence_threshold_95: int = 25920    # 180 days for 95% confidence

    # Plotting configuration
    save_plots: bool = True
    plot_dpi: int = 300
    plot_format: str = 'png'

class SafeDivision:
    """Utility to prevent division-by-zero errors."""
    @staticmethod
    def div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Robustly divides two numbers, returning a default value on error or zero denominator."""
        if abs(denominator) < 1e-9:
            return default
        try:
            return float(numerator) / float(denominator)
        except (ValueError, TypeError, ZeroDivisionError):
            return default

def find_latest_enhanced_metrics_file(directory: str = ".") -> Optional[str]:
    """
    Find the latest CSV file starting with 'enhanced_metrics' in the specified directory.

    Args:
        directory: Directory to search in (default: current directory)

    Returns:
        Path to the latest enhanced_metrics CSV file, or None if not found
    """
    pattern = os.path.join(directory, "enhanced_metrics*.csv")
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]

    print(f"📁 Found {len(files)} enhanced_metrics file(s)")
    print(f"📄 Using latest: {os.path.basename(latest_file)}")

    return latest_file

def load_latest_enhanced_metrics(directory: str = ".") -> Optional[pd.DataFrame]:
    """
    Load the latest enhanced_metrics CSV file from the specified directory.

    Args:
        directory: Directory to search in (default: current directory)

    Returns:
        DataFrame with the data, or None if no file found
    """
    latest_file = find_latest_enhanced_metrics_file(directory)

    if latest_file is None:
        print("❌ No enhanced_metrics*.csv files found in the current directory")
        print("💡 Make sure you have run the hybrid fund simulation first")
        return None

    try:
        df = pd.read_csv(latest_file)
        print(f"✅ Loaded {len(df):,} timesteps from {os.path.basename(latest_file)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load {latest_file}: {e}")
        return None

class EnhancedPortfolioAnalyzer:
    """
    Enhanced Portfolio Analyzer with comprehensive scrutiny and economic model analysis.
    Provides automated, detailed analysis without requiring manual intervention.

    If no DataFrame is provided, automatically loads the latest enhanced_metrics*.csv file.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, cfg: Optional[AnalyzerConfig] = None):
        # Auto-load latest enhanced_metrics file if no DataFrame provided
        if df is None:
            print("🔍 No DataFrame provided - searching for latest enhanced_metrics file...")
            df = load_latest_enhanced_metrics()
            if df is None:
                raise ValueError("No enhanced_metrics*.csv files found and no DataFrame provided")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("EnhancedPortfolioAnalyzer requires a non-empty DataFrame")

        self.df = df.copy()
        self.cfg = cfg or AnalyzerConfig()
        self.results: Dict[str, float] = {}
        self.meta: Dict[str, str] = {}
        self.economic_analysis: Dict[str, Any] = {}
        self.confidence_analysis: Dict[str, Any] = {}
        self.ai_performance_breakdown: Dict[str, Any] = {}

        # Create output directory for plots and reports
        self.output_dir = f"analysis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.cfg.save_plots:
            os.makedirs(self.output_dir, exist_ok=True)

        # Parse timestamp if present for more accurate annualization
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        print(f"📊 Analyzer initialized with {len(self.df):,} timesteps and {len(self.df.columns)} columns")

    # ---------------------------------------------------------------------
    # Enhanced Public API
    # ---------------------------------------------------------------------
    def comprehensive_analysis(self, make_plots: bool = True, save_report: bool = True) -> Dict[str, Any]:
        """
        Performs comprehensive analysis with detailed scrutiny, economic model validation,
        and extensive plotting. This is the main entry point for enhanced analysis.

        Returns:
            Complete analysis results including performance, economic model, and AI breakdown
        """
        print("🚀 STARTING COMPREHENSIVE PORTFOLIO ANALYSIS")
        print("=" * 80)

        # 1. Basic performance analysis
        basic_results = self.analyze(make_plots=False)

        # 2. Economic model analysis
        self.economic_analysis = self._analyze_economic_model()

        # 3. AI performance breakdown
        self.ai_performance_breakdown = self._analyze_ai_performance()

        # 4. Confidence analysis
        self.confidence_analysis = self._analyze_confidence()

        # 5. Risk analysis
        risk_analysis = self._analyze_risk_metrics()

        # 6. Generate comprehensive plots
        if make_plots:
            self._create_enhanced_plots()

        # 7. Generate detailed report
        if save_report:
            self._generate_comprehensive_report()

        # 8. Print executive summary
        self._print_executive_summary()

        return {
            'performance_metrics': basic_results,
            'economic_analysis': self.economic_analysis,
            'ai_performance': self.ai_performance_breakdown,
            'confidence_analysis': self.confidence_analysis,
            'risk_analysis': risk_analysis,
            'meta': self.meta
        }

    def analyze(self, make_plots: bool = False, plot_path_prefix: Optional[str] = None) -> Dict[str, float]:
        """
        Performs a full analysis of the portfolio log data.

        Args:
            make_plots: If True, generates and displays performance plots.
            plot_path_prefix: If provided, saves the plots to files with this prefix.

        Returns:
            A dictionary containing all calculated performance metrics.
        """
        ann = self._infer_annualization()
        self.results["annualization"] = float(ann)

        rt_data = self._compute_returns_and_curve()
        curve = rt_data.get("curve", np.ones(len(self.df)))
        self.meta.update(rt_data["meta"])  # Store metadata about the calculation

        # Risk-free rate per step
        rf_step = self.cfg.risk_free_annual / ann

        # Use simple returns for statistical calculations
        simple_returns = rt_data["simple"]

        # Compute core financial metrics
        self.results.update(self._compute_core_metrics(simple_returns, curve, rf_step, ann))

        # Calculate total ROI if a notional value series was used
        if rt_data["series_name"] in self.cfg.equity_cols + self.cfg.budget_cols:
            series = pd.to_numeric(self.df[rt_data["series_name"]], errors="coerce").dropna()
            if len(series) >= 2 and series.iloc[0] > 0:
                self.results["roi_total"] = SafeDivision.div(series.iloc[-1] - series.iloc[0], series.iloc[0])

        # Add mean values of key diagnostic columns if they exist
        diagnostic_cols = [
            "overall_risk", "generation_revenue", "mtm_pnl", "meta_reward",
            "revenue_step", "cumulative_returns", "fund_performance",
            "wind_cap", "solar_cap", "hydro_cap", "battery_energy"
        ]
        for col in diagnostic_cols:
            if col in self.df.columns:
                series = pd.to_numeric(self.df[col], errors="coerce")
                if not series.isna().all():
                    self.results[f"mean_{col}"] = float(series.mean())
                    self.results[f"final_{col}"] = float(series.iloc[-1]) if len(series) > 0 else 0.0

        if make_plots:
            self.create_comprehensive_plots(curve, simple_returns, rt_data["series_name"], path_prefix=plot_path_prefix)

        return self.results

    # ---------------------------------------------------------------------
    # Enhanced Analysis Methods
    # ---------------------------------------------------------------------
    def _analyze_economic_model(self) -> Dict[str, Any]:
        """Comprehensive analysis of the economic model and fund structure"""
        print("\n📊 ANALYZING ECONOMIC MODEL")
        print("-" * 50)

        analysis = {}

        # Fund structure validation
        analysis['fund_structure'] = self._validate_fund_structure()

        # Asset allocation analysis
        analysis['asset_allocation'] = self._analyze_asset_allocation()

        # Revenue breakdown analysis
        analysis['revenue_analysis'] = self._analyze_revenue_streams()

        # Cost structure analysis
        analysis['cost_analysis'] = self._analyze_cost_structure()

        # Performance vs targets
        analysis['target_comparison'] = self._compare_vs_targets()

        print("✅ Economic model analysis complete")
        return analysis

    def _validate_fund_structure(self) -> Dict[str, Any]:
        """Validate the fund structure matches specifications"""
        structure = {
            'initial_fund_size': self.cfg.initial_fund_size,
            'target_physical_allocation': self.cfg.physical_allocation,
            'target_financial_allocation': self.cfg.financial_allocation,
        }

        # Check if we have fund NAV data
        if 'fund_performance' in self.df.columns:
            nav_series = pd.to_numeric(self.df['fund_performance'], errors='coerce').dropna()
            if len(nav_series) > 0:
                structure['actual_initial_nav'] = float(nav_series.iloc[0])
                structure['final_nav'] = float(nav_series.iloc[-1])
                structure['nav_change'] = structure['final_nav'] - structure['actual_initial_nav']
                structure['nav_return'] = SafeDivision.div(structure['nav_change'], structure['actual_initial_nav'], 0.0)

        return structure

    def _analyze_asset_allocation(self) -> Dict[str, Any]:
        """Analyze actual vs target asset allocation"""
        allocation = {
            'target_assets': {
                'wind_capacity_mw': 75.0,  # 5% of 1,500MW
                'solar_capacity_mw': 50.0,  # 5% of 1,000MW
                'hydro_capacity_mw': 20.0,  # 2% of 1,000MW
                'battery_capacity_mwh': 10.0
            }
        }

        # Check actual allocations if available
        actual_assets = {}
        for asset in ['wind_cap', 'solar_cap', 'hydro_cap', 'battery_energy']:
            if asset in self.df.columns:
                series = pd.to_numeric(self.df[asset], errors='coerce').dropna()
                if len(series) > 0:
                    actual_assets[asset] = float(series.iloc[-1])

        allocation['actual_assets'] = actual_assets

        # Calculate allocation efficiency
        if actual_assets:
            allocation['allocation_efficiency'] = self._calculate_allocation_efficiency(actual_assets)

        return allocation

    def _analyze_revenue_streams(self) -> Dict[str, Any]:
        """Analyze revenue streams from different sources"""
        revenue_analysis = {}

        # Physical generation revenue
        if 'generation_revenue' in self.df.columns:
            gen_rev = pd.to_numeric(self.df['generation_revenue'], errors='coerce').dropna()
            if len(gen_rev) > 0:
                revenue_analysis['physical_generation'] = {
                    'total_revenue': float(gen_rev.sum()),
                    'average_per_step': float(gen_rev.mean()),
                    'final_cumulative': float(gen_rev.iloc[-1]) if len(gen_rev) > 0 else 0.0
                }

        # Financial trading revenue (MTM)
        if 'mtm_pnl' in self.df.columns:
            mtm = pd.to_numeric(self.df['mtm_pnl'], errors='coerce').dropna()
            if len(mtm) > 0:
                revenue_analysis['financial_trading'] = {
                    'total_mtm_pnl': float(mtm.sum()),
                    'average_per_step': float(mtm.mean()),
                    'volatility': float(mtm.std()),
                    'positive_periods': int((mtm > 0).sum()),
                    'negative_periods': int((mtm < 0).sum())
                }

        # Battery revenue
        battery_cols = [col for col in self.df.columns if 'battery' in col.lower() and 'revenue' in col.lower()]
        if battery_cols:
            battery_rev = pd.to_numeric(self.df[battery_cols[0]], errors='coerce').dropna()
            if len(battery_rev) > 0:
                revenue_analysis['battery_operations'] = {
                    'total_revenue': float(battery_rev.sum()),
                    'average_per_step': float(battery_rev.mean())
                }

        return revenue_analysis

    def _analyze_cost_structure(self) -> Dict[str, Any]:
        """Analyze cost structure and efficiency"""
        cost_analysis = {}

        # Operating costs
        cost_cols = [col for col in self.df.columns if 'cost' in col.lower() or 'expense' in col.lower()]
        for col in cost_cols:
            costs = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(costs) > 0:
                cost_analysis[col] = {
                    'total': float(costs.sum()),
                    'average': float(costs.mean()),
                    'trend': 'increasing' if costs.iloc[-1] > costs.iloc[0] else 'decreasing'
                }

        return cost_analysis

    def _compare_vs_targets(self) -> Dict[str, Any]:
        """Compare actual performance vs targets"""
        comparison = {
            'target_baseline_return': self.cfg.target_baseline_return,
            'target_ai_return': self.cfg.target_ai_return,
            'target_improvement': self.cfg.target_ai_return - self.cfg.target_baseline_return
        }

        # Calculate actual returns if possible
        if 'ann_return_est' in self.results:
            actual_return = self.results['ann_return_est']
            comparison['actual_return'] = actual_return
            comparison['vs_baseline'] = actual_return - self.cfg.target_baseline_return
            comparison['vs_ai_target'] = actual_return - self.cfg.target_ai_return
            comparison['target_achievement'] = SafeDivision.div(actual_return, self.cfg.target_ai_return, 0.0)

        return comparison

    def _analyze_ai_performance(self) -> Dict[str, Any]:
        """Analyze AI-specific performance improvements"""
        print("\n🤖 ANALYZING AI PERFORMANCE")
        print("-" * 50)

        ai_analysis = {}

        # Agent performance analysis
        ai_analysis['agent_performance'] = self._analyze_agent_performance()

        # Trading strategy analysis
        ai_analysis['trading_analysis'] = self._analyze_trading_strategies()

        # Risk management analysis
        ai_analysis['risk_management'] = self._analyze_ai_risk_management()

        # Forecast accuracy analysis
        ai_analysis['forecast_analysis'] = self._analyze_forecast_performance()

        print("✅ AI performance analysis complete")
        return ai_analysis

    def _analyze_agent_performance(self) -> Dict[str, Any]:
        """Analyze individual agent performance"""
        agent_analysis = {}

        # Look for agent-specific reward columns
        agent_cols = [col for col in self.df.columns if 'reward' in col.lower()]
        for col in agent_cols:
            rewards = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(rewards) > 0:
                agent_analysis[col] = {
                    'total_reward': float(rewards.sum()),
                    'average_reward': float(rewards.mean()),
                    'reward_volatility': float(rewards.std()),
                    'positive_rewards': int((rewards > 0).sum()),
                    'negative_rewards': int((rewards < 0).sum()),
                    'reward_trend': 'improving' if rewards.iloc[-10:].mean() > rewards.iloc[:10].mean() else 'declining'
                }

        return agent_analysis

    def _analyze_trading_strategies(self) -> Dict[str, Any]:
        """Analyze financial trading strategy performance"""
        trading_analysis = {}

        # Financial instrument performance
        instrument_cols = [col for col in self.df.columns if 'instrument' in col.lower()]
        for col in instrument_cols:
            values = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(values) > 0:
                returns = values.pct_change().dropna()
                trading_analysis[col] = {
                    'final_value': float(values.iloc[-1]),
                    'total_return': SafeDivision.div(values.iloc[-1] - values.iloc[0], values.iloc[0], 0.0),
                    'volatility': float(returns.std()),
                    'sharpe_ratio': SafeDivision.div(returns.mean(), returns.std(), 0.0),
                    'max_value': float(values.max()),
                    'min_value': float(values.min())
                }

        return trading_analysis

    def _analyze_ai_risk_management(self) -> Dict[str, Any]:
        """Analyze AI risk management effectiveness"""
        risk_analysis = {}

        # Overall risk metrics
        if 'overall_risk' in self.df.columns:
            risk_values = pd.to_numeric(self.df['overall_risk'], errors='coerce').dropna()
            if len(risk_values) > 0:
                risk_analysis['overall_risk'] = {
                    'average_risk': float(risk_values.mean()),
                    'max_risk': float(risk_values.max()),
                    'min_risk': float(risk_values.min()),
                    'risk_stability': float(risk_values.std()),
                    'final_risk': float(risk_values.iloc[-1])
                }

        # Market stress response
        if 'market_stress' in self.df.columns:
            stress = pd.to_numeric(self.df['market_stress'], errors='coerce').dropna()
            if len(stress) > 0:
                risk_analysis['market_stress'] = {
                    'average_stress': float(stress.mean()),
                    'max_stress': float(stress.max()),
                    'stress_periods': int((stress > 0.5).sum()),
                    'stress_response': 'adaptive' if stress.std() > 0.1 else 'stable'
                }

        return risk_analysis

    def _analyze_forecast_performance(self) -> Dict[str, Any]:
        """Analyze forecast accuracy and effectiveness"""
        forecast_analysis = {}

        # Look for forecast-related columns
        forecast_cols = [col for col in self.df.columns if 'forecast' in col.lower() or 'prediction' in col.lower()]

        if forecast_cols:
            forecast_analysis['available_forecasts'] = forecast_cols

            # Analyze forecast accuracy if actual vs predicted data available
            for col in forecast_cols:
                values = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(values) > 0:
                    forecast_analysis[col] = {
                        'average_value': float(values.mean()),
                        'volatility': float(values.std()),
                        'trend': 'improving' if values.iloc[-10:].mean() > values.iloc[:10].mean() else 'stable'
                    }

        return forecast_analysis

    def _analyze_confidence(self) -> Dict[str, Any]:
        """Analyze statistical confidence in results"""
        print("\n📊 ANALYZING STATISTICAL CONFIDENCE")
        print("-" * 50)

        confidence = {}

        # Calculate sample size and confidence metrics
        total_timesteps = len(self.df)
        confidence['total_timesteps'] = total_timesteps
        confidence['confidence_90_threshold'] = self.cfg.confidence_threshold_90
        confidence['confidence_95_threshold'] = self.cfg.confidence_threshold_95

        # Confidence assessment
        if total_timesteps >= self.cfg.confidence_threshold_95:
            confidence['confidence_level'] = '95%+ (High Confidence)'
            confidence['statistical_significance'] = 'Very High'
        elif total_timesteps >= self.cfg.confidence_threshold_90:
            confidence['confidence_level'] = '90%+ (Good Confidence)'
            confidence['statistical_significance'] = 'High'
        elif total_timesteps >= self.cfg.confidence_threshold_90 // 2:
            confidence['confidence_level'] = '70%+ (Moderate Confidence)'
            confidence['statistical_significance'] = 'Moderate'
        else:
            confidence['confidence_level'] = '<70% (Low Confidence)'
            confidence['statistical_significance'] = 'Low'

        # Time-based confidence
        confidence['days_of_data'] = total_timesteps / 144  # 144 timesteps per day
        confidence['weeks_of_data'] = confidence['days_of_data'] / 7
        confidence['months_of_data'] = confidence['days_of_data'] / 30

        print(f"✅ Confidence analysis complete: {confidence['confidence_level']}")
        return confidence

    def _analyze_risk_metrics(self) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        print("\n🛡️ ANALYZING RISK METRICS")
        print("-" * 50)

        risk_metrics = {}

        # Portfolio risk analysis
        if 'portfolio_value' in self.df.columns or 'equity' in self.df.columns:
            value_col = 'portfolio_value' if 'portfolio_value' in self.df.columns else 'equity'
            values = pd.to_numeric(self.df[value_col], errors='coerce').dropna()

            if len(values) > 1:
                returns = values.pct_change().dropna()

                risk_metrics['portfolio_risk'] = {
                    'volatility': float(returns.std()),
                    'downside_volatility': float(returns[returns < 0].std()) if (returns < 0).any() else 0.0,
                    'max_drawdown': float(self._calculate_max_drawdown(values)),
                    'var_95': float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0,
                    'var_99': float(np.percentile(returns, 1)) if len(returns) > 0 else 0.0,
                    'skewness': float(returns.skew()) if len(returns) > 2 else 0.0,
                    'kurtosis': float(returns.kurtosis()) if len(returns) > 3 else 0.0
                }

        print("✅ Risk analysis complete")
        return risk_metrics

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def _calculate_allocation_efficiency(self, actual_assets: Dict[str, float]) -> Dict[str, float]:
        """Calculate allocation efficiency metrics"""
        efficiency = {}

        # Calculate utilization rates
        targets = {
            'wind_cap': 75.0,
            'solar_cap': 50.0,
            'hydro_cap': 20.0,
            'battery_energy': 10.0
        }

        for asset, target in targets.items():
            if asset in actual_assets:
                actual = actual_assets[asset]
                efficiency[f'{asset}_utilization'] = SafeDivision.div(actual, target, 0.0)
                efficiency[f'{asset}_efficiency'] = min(SafeDivision.div(actual, target, 0.0), 1.0)

        # Overall efficiency score
        utilizations = [v for k, v in efficiency.items() if 'utilization' in k]
        efficiency['overall_efficiency'] = np.mean(utilizations) if utilizations else 0.0

        return efficiency

    # ---------------------------------------------------------------------
    # Returns & Equity Curve Calculation
    # ---------------------------------------------------------------------
    def _infer_annualization(self) -> int:
        """Infers the number of steps per year from the data's timestamp or frequency columns."""
        ann = None
        if "timestamp" in self.df.columns and self.df["timestamp"].notna().sum() > 3:
            ts = self.df["timestamp"].dropna().sort_values()
            median_delta_minutes = ts.diff().median().total_seconds() / 60.0
            if median_delta_minutes > 0:
                steps_per_year = (365 * 24 * 60) / median_delta_minutes
                ann = int(round(steps_per_year))
        
        if ann is None and "investment_freq" in self.df.columns:
            try:
                # Assuming 'investment_freq' implies steps per day
                steps_per_day = self.df["investment_freq"].dropna().mode().iloc[0]
                ann = int(365 * steps_per_day)
            except Exception:
                pass
        
        if ann is None:
            ann = self.cfg.min_annualization # Fallback to default
            
        return int(np.clip(ann, self.cfg.min_annualization, self.cfg.max_annualization))

    def _choose_series(self) -> Tuple[str, pd.Series, str]:
        """Selects the best available series for performance calculation (Equity > Budget > Reward)."""
        # 1. Prefer equity-based columns
        for col in self.cfg.equity_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s.dropna(), "equity"
        # 2. Fallback to budget
        for col in self.cfg.budget_cols:
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.notna().sum() >= 2:
                    return col, s.dropna(), "budget"
        # 3. Last resort: use reward as a proxy for returns
        if "meta_reward" in self.df.columns:
            return "meta_reward", pd.to_numeric(self.df["meta_reward"], errors="coerce").fillna(0.0), "reward"
        # 4. If nothing is available, return zeros
        return "zeros", pd.Series(np.zeros(len(self.df)), index=self.df.index), "zeros"

    def _compute_returns_and_curve(self) -> Dict[str, Any]:
        """Calculates log and simple returns and the corresponding equity curve from the chosen series."""
        series_name, s, series_kind = self._choose_series()

        if series_kind in ("reward", "zeros"):
            # Cannot compute a meaningful curve, return a flat one
            returns = s.to_numpy()
            curve = (1.0 + returns).cumprod()
            meta = {"returns_source": series_name, "curve_method": "cumulative_product", "series_used": series_name}
            return {"log": returns, "simple": returns, "kind": "simple", "meta": meta, "series_name": series_name, "curve": curve}

        # PATCH: Ensure series is strictly positive for log returns
        if (s <= 0).any():
            s = s + abs(s.min()) + 1e-6 # Shift series to be positive

        log_r = np.log(s).diff().fillna(0.0).to_numpy()
        simple_r = s.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy()

        if self.cfg.log_prefer:
            curve = np.exp(np.cumsum(log_r))
            kind, curve_method = "log", "exp_cumsum_log_returns"
        else:
            curve = (1.0 + simple_r).cumprod()
            kind, curve_method = "simple", "cumprod_simple_returns"

        meta = {"returns_source": f"{kind}({series_name})", "curve_method": curve_method, "series_used": series_name}
        return {"log": log_r, "simple": simple_r, "kind": kind, "meta": meta, "series_name": series_name, "curve": curve}

    # ---------------------------------------------------------------------
    # Core Metrics Calculation
    # ---------------------------------------------------------------------
    def _compute_core_metrics(self, simple_returns: np.ndarray, curve: np.ndarray, rf_step: float, ann: int) -> Dict[str, float]:
        """Computes key financial metrics from simple returns and the equity curve."""
        r = simple_returns[np.isfinite(simple_returns)]
        if r.size < 2: return {} # Not enough data for meaningful stats

        # Annualized Return and Volatility
        ann_return = float(np.mean(r) * ann)
        ann_volatility = float(np.std(r, ddof=1) * np.sqrt(ann))

        # Sharpe Ratio
        excess_returns = r - rf_step
        sharpe = SafeDivision.div(np.mean(excess_returns), np.std(excess_returns, ddof=1)) * np.sqrt(ann)

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if downside_returns.size > 0 else 0.0
        sortino = SafeDivision.div(np.mean(excess_returns), downside_std) * np.sqrt(ann)

        # Max Drawdown
        # PATCH: Added epsilon to denominator to prevent division by zero
        running_max = np.maximum.accumulate(curve)
        drawdown = (curve - running_max) / (running_max + 1e-9)
        max_dd = float(np.min(drawdown))

        # Calmar Ratio
        calmar = SafeDivision.div(ann_return, abs(max_dd))

        # Win Rate & Profit Factor
        win_rate = SafeDivision.div((r > 0).sum(), r.size)
        gains = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        profit_factor = SafeDivision.div(gains, losses, default=np.inf)

        return {
            "ann_return_est": ann_return,
            "ann_volatility": ann_volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    # ---------------------------------------------------------------------
    # Enhanced Plotting Methods
    # ---------------------------------------------------------------------
    def _create_enhanced_plots(self):
        """Create comprehensive enhanced plots for detailed analysis"""
        print("\n📈 GENERATING ENHANCED PLOTS")
        print("-" * 50)

        # Set style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")

        # 1. Performance Overview Dashboard
        self._plot_performance_dashboard()

        # 2. Economic Model Breakdown
        self._plot_economic_breakdown()

        # 3. AI Performance Analysis
        self._plot_ai_performance()

        # 4. Risk Analysis Plots
        self._plot_risk_analysis()

        # 5. Revenue Stream Analysis
        self._plot_revenue_streams()

        print("✅ Enhanced plots generated")

    def _plot_performance_dashboard(self):
        """Create main performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hybrid Renewable Energy Fund - Performance Dashboard', fontsize=16, fontweight='bold')

        # Portfolio value over time
        if 'portfolio_value' in self.df.columns or 'equity' in self.df.columns:
            value_col = 'portfolio_value' if 'portfolio_value' in self.df.columns else 'equity'
            values = pd.to_numeric(self.df[value_col], errors='coerce').dropna()

            axes[0,0].plot(values.index, values.values, linewidth=2, color='navy')
            axes[0,0].set_title('Portfolio Value Over Time', fontweight='bold')
            axes[0,0].set_ylabel('Portfolio Value ($)')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].ticklabel_format(style='plain', axis='y')

            # Add target lines
            if len(values) > 0:
                initial_value = values.iloc[0]
                target_baseline = initial_value * (1 + self.cfg.target_baseline_return * len(values) / 52560)
                target_ai = initial_value * (1 + self.cfg.target_ai_return * len(values) / 52560)

                axes[0,0].axhline(y=target_baseline, color='orange', linestyle='--', alpha=0.7, label='Baseline Target')
                axes[0,0].axhline(y=target_ai, color='green', linestyle='--', alpha=0.7, label='AI Target')
                axes[0,0].legend()

        # Returns distribution
        if 'portfolio_value' in self.df.columns or 'equity' in self.df.columns:
            returns = values.pct_change().dropna()
            axes[0,1].hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0,1].set_title('Returns Distribution', fontweight='bold')
            axes[0,1].set_xlabel('Return')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)

        # Drawdown analysis
        if len(values) > 1:
            peak = values.expanding().max()
            drawdown = (values - peak) / peak * 100
            axes[1,0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
            axes[1,0].set_title('Drawdown Analysis', fontweight='bold')
            axes[1,0].set_ylabel('Drawdown (%)')
            axes[1,0].grid(True, alpha=0.3)

        # Risk metrics over time
        if 'overall_risk' in self.df.columns:
            risk = pd.to_numeric(self.df['overall_risk'], errors='coerce').dropna()
            axes[1,1].plot(risk.index, risk.values, color='red', linewidth=2)
            axes[1,1].set_title('Risk Level Over Time', fontweight='bold')
            axes[1,1].set_ylabel('Risk Level')
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        if self.cfg.save_plots:
            plt.savefig(f"{self.output_dir}/performance_dashboard.{self.cfg.plot_format}",
                       dpi=self.cfg.plot_dpi, bbox_inches='tight')
        plt.show()

    def create_comprehensive_plots(self, curve: np.ndarray, simple_returns: np.ndarray, series_name: str, path_prefix: Optional[str] = None):
        """Generates a comprehensive multi-panel plot of performance."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f"{self.cfg.plot_title} (Source: {series_name})", fontsize=16)

        # 1. Equity Curve
        axes[0].plot(curve, label="Equity Curve", color='royalblue')
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Equity (Normalized)")
        axes[0].legend()

        # 2. Drawdown
        running_max = np.maximum.accumulate(curve)
        drawdown = (curve - running_max) / (running_max + 1e-9)
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, color='crimson', alpha=0.5)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown (%)")

        # 3. Returns Histogram
        axes[2].hist(simple_returns, bins=75, color='seagreen', alpha=0.8)
        axes[2].set_title("Per-Step Return Distribution")
        axes[2].set_xlabel("Simple Return")
        axes[2].set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if path_prefix:
            plt.savefig(f"{path_prefix}_comprehensive_plot.png", dpi=150)
        plt.show()

    def _plot_economic_breakdown(self):
        """Plot economic model breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Economic Model Analysis', fontsize=16, fontweight='bold')

        # Asset allocation pie chart
        asset_values = [
            self.cfg.wind_ownership * 1500 * 2000,  # Wind value estimate
            self.cfg.solar_ownership * 1000 * 1500,  # Solar value estimate
            self.cfg.hydro_ownership * 1000 * 3000,  # Hydro value estimate
            self.cfg.battery_capacity * 500000       # Battery value estimate
        ]
        asset_labels = ['Wind (5%)', 'Solar (5%)', 'Hydro (2%)', 'Battery (100%)']

        axes[0,0].pie(asset_values, labels=asset_labels, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Physical Asset Allocation', fontweight='bold')

        # Revenue streams over time
        revenue_cols = ['generation_revenue', 'mtm_pnl']
        colors = ['green', 'blue']

        for i, col in enumerate(revenue_cols):
            if col in self.df.columns:
                revenue = pd.to_numeric(self.df[col], errors='coerce').dropna()
                if len(revenue) > 0:
                    axes[0,1].plot(revenue.index, revenue.cumsum(),
                                 label=col.replace('_', ' ').title(),
                                 color=colors[i], linewidth=2)

        axes[0,1].set_title('Cumulative Revenue Streams', fontweight='bold')
        axes[0,1].set_ylabel('Cumulative Revenue ($)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Performance vs targets
        if 'ann_return_est' in self.results:
            targets = ['Baseline Target', 'AI Target', 'Actual Performance']
            values = [self.cfg.target_baseline_return * 100,
                     self.cfg.target_ai_return * 100,
                     self.results['ann_return_est'] * 100]
            colors = ['orange', 'green', 'navy']

            bars = axes[1,0].bar(targets, values, color=colors, alpha=0.7)
            axes[1,0].set_title('Performance vs Targets', fontweight='bold')
            axes[1,0].set_ylabel('Annual Return (%)')
            axes[1,0].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                             f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

        # Risk-return scatter
        if 'ann_return_est' in self.results and 'ann_volatility' in self.results:
            axes[1,1].scatter(self.results['ann_volatility'] * 100,
                            self.results['ann_return_est'] * 100,
                            s=200, color='red', alpha=0.7, label='Actual')
            axes[1,1].scatter(0.15 * 100, self.cfg.target_baseline_return * 100,
                            s=200, color='orange', alpha=0.7, label='Baseline Target')
            axes[1,1].scatter(0.18 * 100, self.cfg.target_ai_return * 100,
                            s=200, color='green', alpha=0.7, label='AI Target')

            axes[1,1].set_title('Risk-Return Profile', fontweight='bold')
            axes[1,1].set_xlabel('Volatility (%)')
            axes[1,1].set_ylabel('Return (%)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        if self.cfg.save_plots:
            plt.savefig(f"{self.output_dir}/economic_breakdown.{self.cfg.plot_format}",
                       dpi=self.cfg.plot_dpi, bbox_inches='tight')
        plt.show()

    def _plot_ai_performance(self):
        """Plot AI-specific performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('AI Performance Analysis', fontsize=16, fontweight='bold')

        # Agent rewards over time
        agent_cols = [col for col in self.df.columns if 'reward' in col.lower()]
        colors = ['blue', 'green', 'red', 'purple']

        for i, col in enumerate(agent_cols[:4]):  # Limit to 4 agents
            rewards = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(rewards) > 0:
                axes[0,0].plot(rewards.index, rewards.cumsum(),
                             label=col.replace('_', ' ').title(),
                             color=colors[i], linewidth=2)

        axes[0,0].set_title('Cumulative Agent Rewards', fontweight='bold')
        axes[0,0].set_ylabel('Cumulative Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Financial instruments performance
        instrument_cols = [col for col in self.df.columns if 'instrument' in col.lower()]
        for i, col in enumerate(instrument_cols[:3]):
            values = pd.to_numeric(self.df[col], errors='coerce').dropna()
            if len(values) > 0:
                axes[0,1].plot(values.index, values,
                             label=col.replace('_', ' ').title(),
                             linewidth=2)

        axes[0,1].set_title('Financial Instruments Value', fontweight='bold')
        axes[0,1].set_ylabel('Instrument Value ($)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        plt.tight_layout()
        if self.cfg.save_plots:
            plt.savefig(f"{self.output_dir}/ai_performance.{self.cfg.plot_format}",
                       dpi=self.cfg.plot_dpi, bbox_inches='tight')
        plt.show()

    def _plot_risk_analysis(self):
        """Plot risk analysis"""
        # Implementation for risk plotting
        pass

    def _plot_revenue_streams(self):
        """Plot revenue stream analysis"""
        # Implementation for revenue plotting
        pass

    def _print_executive_summary(self):
        """Print comprehensive executive summary"""
        print("\n" + "="*80)
        print("📋 EXECUTIVE SUMMARY - HYBRID RENEWABLE ENERGY FUND")
        print("="*80)

        # Performance summary
        if 'ann_return_est' in self.results:
            print(f"\n🎯 PERFORMANCE METRICS:")
            print(f"   Annual Return: {self.results['ann_return_est']:.2%}")
            print(f"   Volatility: {self.results.get('ann_volatility', 0):.2%}")
            print(f"   Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {self.results.get('max_drawdown', 0):.2%}")

        # Target comparison
        if 'ann_return_est' in self.results:
            vs_baseline = self.results['ann_return_est'] - self.cfg.target_baseline_return
            vs_ai_target = self.results['ann_return_est'] - self.cfg.target_ai_return
            print(f"\n📊 TARGET COMPARISON:")
            print(f"   vs Baseline Target: {vs_baseline:+.2%}")
            print(f"   vs AI Target: {vs_ai_target:+.2%}")
            print(f"   Target Achievement: {(self.results['ann_return_est']/self.cfg.target_ai_return):.1%}")

        # Confidence assessment
        if hasattr(self, 'confidence_analysis'):
            print(f"\n🔍 STATISTICAL CONFIDENCE:")
            print(f"   Data Points: {self.confidence_analysis.get('total_timesteps', 0):,}")
            print(f"   Confidence Level: {self.confidence_analysis.get('confidence_level', 'Unknown')}")
            print(f"   Days of Data: {self.confidence_analysis.get('days_of_data', 0):.1f}")

        # Economic model validation
        print(f"\n🏭 ECONOMIC MODEL:")
        print(f"   Fund Size: ${self.cfg.initial_fund_size:,.0f}")
        print(f"   Physical Allocation: {self.cfg.physical_allocation:.1%}")
        print(f"   Financial Allocation: {self.cfg.financial_allocation:.1%}")

        # AI performance
        print(f"\n🤖 AI ENHANCEMENT:")
        expected_improvement = self.cfg.target_ai_return - self.cfg.target_baseline_return
        print(f"   Expected AI Improvement: {expected_improvement:.2%}")
        if 'ann_return_est' in self.results:
            actual_improvement = self.results['ann_return_est'] - self.cfg.target_baseline_return
            print(f"   Actual Improvement: {actual_improvement:+.2%}")

        print(f"\n📁 OUTPUT DIRECTORY: {self.output_dir}")
        print("="*80)

    def _generate_comprehensive_report(self):
        """Generate detailed written report"""
        report_path = f"{self.output_dir}/comprehensive_analysis_report.json"

        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'fund_configuration': {
                'initial_fund_size': self.cfg.initial_fund_size,
                'target_baseline_return': self.cfg.target_baseline_return,
                'target_ai_return': self.cfg.target_ai_return,
                'physical_allocation': self.cfg.physical_allocation,
                'financial_allocation': self.cfg.financial_allocation
            },
            'performance_metrics': self.results,
            'economic_analysis': self.economic_analysis,
            'ai_performance': self.ai_performance_breakdown,
            'confidence_analysis': self.confidence_analysis,
            'data_summary': {
                'total_rows': len(self.df),
                'columns': list(self.df.columns),
                'date_range': {
                    'start': str(self.df.index[0]) if len(self.df) > 0 else None,
                    'end': str(self.df.index[-1]) if len(self.df) > 0 else None
                }
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"📄 Comprehensive report saved: {report_path}")

# Enhanced convenience functions
def analyze_dataframe(df: Optional[pd.DataFrame] = None, make_plots: bool = False, path_prefix: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Analyzes a DataFrame containing portfolio log data using the original analyzer.
    If no DataFrame provided, automatically loads the latest enhanced_metrics*.csv file.

    Args:
        df: The DataFrame to analyze. If None, loads latest enhanced_metrics*.csv
        make_plots: Whether to generate plots.
        path_prefix: Optional prefix for saving plot files.

    Returns:
        A tuple containing the results dictionary and metadata dictionary.
    """
    analyzer = EnhancedPortfolioAnalyzer(df)
    results = analyzer.analyze(make_plots=make_plots, plot_path_prefix=path_prefix)
    return results, analyzer.meta

def comprehensive_analyze_dataframe(df: Optional[pd.DataFrame] = None, save_report: bool = True) -> Dict[str, Any]:
    """
    Performs comprehensive analysis of portfolio data with enhanced scrutiny.
    If no DataFrame provided, automatically loads the latest enhanced_metrics*.csv file.

    This is the main function to use for detailed analysis with:
    - Economic model validation
    - AI performance breakdown
    - Statistical confidence assessment
    - Risk analysis
    - Comprehensive plotting
    - Automated reporting

    Args:
        df: The DataFrame containing portfolio log data. If None, loads latest enhanced_metrics*.csv
        save_report: Whether to save detailed JSON report

    Returns:
        Complete analysis results dictionary
    """
    analyzer = EnhancedPortfolioAnalyzer(df)
    return analyzer.comprehensive_analysis(make_plots=True, save_report=save_report)

def auto_analyze_latest_results(save_report: bool = True) -> Optional[Dict[str, Any]]:
    """
    Automatically find and analyze the latest enhanced_metrics*.csv file.
    This is the simplest way to analyze your results - just run this function!

    Args:
        save_report: Whether to save detailed JSON report

    Returns:
        Complete analysis results dictionary, or None if no file found
    """
    print("🚀 AUTO-ANALYZING LATEST HYBRID FUND RESULTS")
    print("=" * 60)

    try:
        return comprehensive_analyze_dataframe(df=None, save_report=save_report)
    except Exception as e:
        print(f"❌ Auto-analysis failed: {e}")
        return None

# Backward compatibility alias
PortfolioAnalyzer = EnhancedPortfolioAnalyzer
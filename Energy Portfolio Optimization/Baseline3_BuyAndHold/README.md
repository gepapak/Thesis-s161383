# Buy-and-Hold Strategy Baseline (Baseline3)

This directory contains the Buy-and-Hold baseline strategy for renewable energy investment evaluation. This baseline represents the simplest possible investment approach: earning the risk-free rate through government bonds or cash equivalents.

## Overview

The Buy-and-Hold baseline serves as the **"floor" for investment performance** - any active strategy should be able to beat this risk-free rate to justify the additional complexity and risk.

### Key Characteristics

- **No active trading** or portfolio rebalancing
- **No market exposure** to renewable energy markets
- **Risk-free return** (Danish government bonds ~2% annually)
- **Zero volatility** (risk-free investment)
- **Zero drawdown** (capital preservation)
- **Compound interest** growth over time

### Strategy Logic

1. **Hold initial capital** in risk-free assets (government bonds/cash)
2. **Earn steady risk-free rate** (2% annually)
3. **Compound returns** over time with no active decisions
4. **Maintain capital preservation** with zero risk

## Implementation

### Core Components

- `buy_and_hold_optimizer.py` - Core optimizer implementing risk-free rate strategy
- `run_buy_and_hold_baseline.py` - Runner script for evaluation
- `results/` - Output directory for results and reports

### Key Features

- **Auto-detect timebase** from timestamp data
- **Compound interest calculation** with precise timestep rates
- **Comprehensive reporting** with performance metrics and visualizations
- **Consistent structure** matching Baseline1 and Baseline2

## Usage

### Command Line

```bash
# Basic usage with auto-detected timebase
python Baseline3_BuyAndHold/run_buy_and_hold_baseline.py \
  --data_path evaluation_dataset/unseendata.csv

# Specify output directory
python Baseline3_BuyAndHold/run_buy_and_hold_baseline.py \
  --data_path evaluation_dataset/unseendata.csv \
  --output_dir Baseline3_BuyAndHold/results

# Limit timesteps for testing
python Baseline3_BuyAndHold/run_buy_and_hold_baseline.py \
  --data_path evaluation_dataset/unseendata.csv \
  --timesteps 1000

# Specify timebase explicitly
python Baseline3_BuyAndHold/run_buy_and_hold_baseline.py \
  --data_path evaluation_dataset/unseendata.csv \
  --timebase_hours 0.1667
```

### Programmatic Usage

```python
from buy_and_hold_optimizer import BuyAndHoldOptimizer

# Initialize optimizer
optimizer = BuyAndHoldOptimizer(
    initial_budget_usd=800_000_000,  # $800M
    timebase_hours=0.1667            # 10-minute intervals
)

# Run single timestep
result = optimizer.step(data_row)

# Get performance metrics
metrics = optimizer.get_performance_metrics()
```

## Expected Performance

### Theoretical Performance

- **Annual Return**: ~2.0% (Danish government bond rate)
- **Volatility**: ~0.0% (risk-free investment)
- **Sharpe Ratio**: 0.0 (no excess return over risk-free rate)
- **Max Drawdown**: 0.0% (capital preservation)

### Example Results (181 days)

```
Initial Portfolio:  $800.0M USD
Final Portfolio:    $808.0M USD
Total Return:       +1.00%
Annual Return:      +2.02%
Volatility:         0.00%
Max Drawdown:       0.00%
Sharpe Ratio:       0.00
```

## Risk Profile

- **Risk Level**: Zero (government bonds)
- **Market Exposure**: None (no renewable energy exposure)
- **Volatility**: Minimal (risk-free rate fluctuations only)
- **Drawdown Risk**: None (capital preservation guaranteed)
- **Liquidity**: High (government bonds/cash)

## Comparison with Other Baselines

| Baseline | Strategy Type | Expected Annual Return | Risk Level | Complexity |
|----------|---------------|----------------------|------------|------------|
| **Baseline1** | Traditional Portfolio | Variable (-50% to +5%) | High | Medium |
| **Baseline2** | Rule-Based Heuristic | +1% to +5% | Medium | Low |
| **Baseline3** | **Buy-and-Hold** | **+2.0%** | **Zero** | **Minimal** |

### Baseline3 Purpose

- **Floor baseline**: Minimum return any strategy should achieve
- **Risk-free benchmark**: Pure capital preservation approach
- **Complexity justification**: Active strategies must beat this to justify effort
- **Academic standard**: Common baseline in finance literature

## Output Structure

Results are saved in the `results/` directory:

```
results/
├── detailed_results.csv          # Per-timestep portfolio evolution
├── summary_metrics.json          # Comprehensive performance metrics
├── summary.json                  # Summary metrics (compatibility)
└── performance_report.png        # Performance visualization
```

### Key Metrics

- **Portfolio Evolution**: Timestep-by-timestep value growth
- **Risk Metrics**: Volatility, drawdown, Sharpe ratio (all ~0)
- **Return Metrics**: Total return, annualized return
- **Strategy Metrics**: Active decisions (0), market exposure (0%)

## Technical Details

### Risk-Free Rate Calculation

```python
# Danish government bond rate
annual_risk_free_rate = 0.02  # 2% annually

# Convert to per-timestep rate
steps_per_year = 8760 / timebase_hours
timestep_rate = annual_risk_free_rate / steps_per_year

# Compound interest
portfolio_value *= (1 + timestep_rate)
```

### Timebase Auto-Detection

```python
# Detect from timestamp column
timestamps = pd.to_datetime(data['timestamp'])
time_deltas = timestamps.diff().dropna()
median_delta_minutes = time_deltas.median().total_seconds() / 60
timebase_hours = median_delta_minutes / 60
```

### Currency Conversion

- **Primary**: USD (for consistency with other baselines)
- **Secondary**: DKK (for local market context)
- **Rate**: 0.145 DKK/USD (from config)

## Validation

### Expected Behavior

1. **Steady growth**: Portfolio value increases smoothly
2. **Predictable returns**: Matches theoretical risk-free rate
3. **Zero volatility**: No fluctuations in returns
4. **No drawdowns**: Portfolio value never decreases

### Validation Checks

```python
# Theoretical vs actual return check
theoretical_annual = (1 + 0.02) ** years_elapsed - 1
actual_annual = metrics['annual_return']
assert abs(theoretical_annual - actual_annual) < 0.001  # Within 0.1%
```

## Integration

### Evaluation Framework

This baseline integrates with the main evaluation framework:

```python
# In evaluation.py
from Baseline3_BuyAndHold.run_buy_and_hold_baseline import BuyAndHoldBaselineRunner

runner = BuyAndHoldBaselineRunner(data_path, output_dir)
runner.run_optimization()
results = runner.get_final_metrics()
```

### Comparison Framework

Results are compatible with baseline comparison tools and can be directly compared with Baseline1 (Traditional Portfolio) and Baseline2 (Rule-Based Heuristic).

## Academic Context

### Finance Literature

The Buy-and-Hold strategy is a standard baseline in academic finance:

- **Capital Asset Pricing Model (CAPM)**: Risk-free rate as baseline
- **Portfolio Theory**: Minimum return for risk-adjusted performance
- **Performance Evaluation**: Sharpe ratio denominator
- **Benchmark Studies**: Common passive strategy comparison

### Renewable Energy Context

In renewable energy investment evaluation:

- **Conservative baseline**: Represents risk-averse investor approach
- **Opportunity cost**: What investor gives up by not investing actively
- **Policy baseline**: Government bond alternative to renewable subsidies
- **Risk comparison**: Highlights risk premium of active strategies

## Limitations

### What This Baseline Does NOT Capture

- **Renewable energy market dynamics**: No exposure to generation or prices
- **Active management value**: No trading or optimization decisions
- **Market timing**: No response to market conditions
- **Portfolio diversification**: Single asset class (bonds/cash)
- **Inflation protection**: Fixed nominal rate (not real rate)

### Appropriate Use Cases

- ✅ **Minimum performance floor**: Any strategy should beat this
- ✅ **Risk-free benchmark**: For Sharpe ratio calculations
- ✅ **Conservative comparison**: Against risk-averse alternatives
- ✅ **Academic baseline**: Standard in finance literature

### Inappropriate Use Cases

- ❌ **Realistic investment alternative**: Too conservative for most investors
- ❌ **Market exposure comparison**: No renewable energy exposure
- ❌ **Active strategy evaluation**: Doesn't test active management skills
- ❌ **Inflation-adjusted returns**: Nominal rate only

## Conclusion

Baseline3 (Buy-and-Hold) provides a **solid foundation** for evaluating renewable energy investment strategies. Any AI agent or active strategy that cannot beat the risk-free rate should be questioned, while strategies that significantly outperform this baseline demonstrate clear value creation.

This baseline ensures that the evaluation framework has a **realistic floor** and helps distinguish between strategies that add genuine value versus those that simply benefit from market exposure.

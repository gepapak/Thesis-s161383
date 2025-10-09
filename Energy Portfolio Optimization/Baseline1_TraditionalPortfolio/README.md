# Baseline1: Improved Traditional Portfolio Optimizer

## Overview

This is an **improved version** of the Traditional Portfolio Optimizer baseline that uses **actual renewable energy data** for realistic performance evaluation.

### Key Improvements

✅ **Uses Real Revenue Data**
- Calculates revenue from renewable generation: `revenue = quantity × price`
- Wind revenue: `wind_generation × electricity_price`
- Solar revenue: `solar_generation × electricity_price`
- Hydro revenue: `hydro_generation × electricity_price`

✅ **Proper Return Calculations**
- Variation-of-return (VoR) for generation assets (robust to near-zero values)
- Log returns for price index
- Risk-free rate for cash holdings

✅ **Multiple Optimization Methods**
- **Equal Weight**: Simple 1/N allocation
- **Minimum Variance**: Minimize portfolio volatility
- **Maximum Sharpe**: Maximize risk-adjusted returns
- **Markowitz Mean-Variance**: Classic MPT with risk aversion
- **Risk Parity**: Equal risk contribution from each asset

✅ **Realistic Portfolio Dynamics**
- Rolling window estimation (252 steps ≈ 1 year)
- Periodic rebalancing (144 steps = 1 day)
- Ledoit-Wolf covariance shrinkage
- No unrealistic constraints or artificial returns

---

## Files

### Core Implementation

1. **`traditional_portfolio_optimizer_v2.py`** - Improved optimizer class
   - `TraditionalPortfolioOptimizer`: Main optimizer
   - `Timebase`: Time configuration (10-minute steps)
   - `OptimizerConfig`: Optimization parameters
   - Asset return calculation from renewable energy data
   - Multiple optimization methods

2. **`run_traditional_baseline_v2.py`** - Improved runner script
   - Loads evaluation data
   - Pre-computes asset returns
   - Runs optimization loop
   - Saves results in standard format

### Legacy Files (Old Implementation)

- `traditional_portfolio_optimizer.py` - Old version (uses random returns)
- `run_traditional_baseline.py` - Old runner (not recommended)

---

## Usage

### Quick Test (1000 steps)

```bash
python test_improved_baseline1.py
```

This will test all optimization methods with 1000 timesteps.

### Full Evaluation (10,000 steps)

```bash
python Baseline1_TraditionalPortfolio/run_traditional_baseline_v2.py \
  --data_path evaluation_dataset/unseendata.csv \
  --timesteps 10000 \
  --method markowitz_mean_variance \
  --output_dir Baseline1_TraditionalPortfolio/results
```

### Available Methods

- `equal_weight` - Equal allocation to all assets
- `min_variance` - Minimum variance portfolio
- `max_sharpe` - Maximum Sharpe ratio
- `markowitz_mean_variance` - Mean-variance optimization (default)
- `risk_parity` - Risk parity allocation

### Command-Line Arguments

- `--data_path`: Path to evaluation data CSV (required)
- `--timesteps`: Number of timesteps to evaluate (default: all data)
- `--method`: Optimization method (default: `markowitz_mean_variance`)
- `--output_dir`: Output directory for results (default: `results`)

---

## How It Works

### 1. Asset Return Calculation

The optimizer calculates returns from actual renewable energy data:

```python
# Revenue from generation (DKK)
wind_revenue = wind_generation × electricity_price
solar_revenue = solar_generation × electricity_price
hydro_revenue = hydro_generation × electricity_price

# Variation-of-return (robust to near-zero values)
return_t = (revenue_t - revenue_{t-1}) / (0.5 × (|revenue_t| + |revenue_{t-1}|) + ε)

# Price index: log return → arithmetic return
price_return = exp(log(price_t / price_{t-1})) - 1

# Cash: risk-free rate
cash_return = (1 + rf_annual)^(Δt / year) - 1
```

### 2. Portfolio Optimization

Every `rebalance_freq` steps (default: 144 = 1 day):

1. **Collect historical returns** (lookback window = 252 steps ≈ 1 year)
2. **Estimate statistics**:
   - Mean returns: `μ = E[r]`
   - Covariance matrix: `Σ = Cov(r)` with Ledoit-Wolf shrinkage
3. **Optimize weights** using selected method
4. **Rebalance portfolio** to new weights

### 3. Portfolio Execution

At each timestep:

1. Calculate current asset returns
2. Compute portfolio return: `r_p = Σ w_i × r_i + w_cash × r_f`
3. Update portfolio value: `V_t = V_{t-1} × (1 + r_p)`
4. Track performance metrics

---

## Output Files

### 1. `detailed_results.csv`

Timestep-by-timestep results:
- `timestep`: Current step
- `portfolio_value`: Portfolio value (USD)
- `returns`: Portfolio return
- `total_return`: Cumulative return
- `sharpe_ratio`: Current Sharpe ratio
- `max_drawdown`: Maximum drawdown
- `volatility`: Portfolio volatility
- `weight_wind`, `weight_solar`, `weight_hydro`, `weight_price`, `weight_cash`: Asset weights

### 2. `summary_metrics.json`

Final performance summary:
```json
{
  "final_value_usd": 843500000.0,
  "initial_value_usd": 800000000.0,
  "total_return": 0.0544,
  "sharpe_ratio": 1.234,
  "max_drawdown": -0.0876,
  "volatility": 0.1234,
  "calmar_ratio": 0.621,
  "optimization_method": "markowitz_mean_variance"
}
```

### 3. `evaluation_results.json`

Standard format for comparison with other baselines:
```json
{
  "method": "Traditional Portfolio - markowitz_mean_variance",
  "total_return": 0.0544,
  "sharpe_ratio": 1.234,
  "max_drawdown": -0.0876,
  "volatility": 0.1234,
  "final_value_usd": 843500000.0,
  "initial_value_usd": 800000000.0,
  "status": "completed"
}
```

---

## Integration with Main Evaluation

The improved baseline is compatible with the main evaluation script:

```bash
python evaluation.py --mode compare \
  --trained_agents Normal/final_models \
  --eval_data evaluation_dataset/unseendata.csv \
  --analyze --plot
```

This will:
1. Run Normal models (AI agents)
2. Run all 3 traditional baselines (including improved Baseline1)
3. Compare performance
4. Generate analysis and plots

---

## Optimization Methods Explained

### Equal Weight
- Simple 1/N allocation
- No optimization required
- Good baseline for comparison

### Minimum Variance
- Minimize: `σ²_p = w^T Σ w`
- Subject to: `Σw = 1`, `w ≥ 0`
- Focuses purely on risk reduction

### Maximum Sharpe
- Maximize: `(μ_p - r_f) / σ_p`
- Subject to: `Σw = 1`, `w ≥ 0`
- Optimal risk-adjusted returns

### Markowitz Mean-Variance
- Minimize: `0.5 × λ × w^T Σ w - (μ - r_f)^T w`
- Subject to: `Σw = 1`, `w ≥ 0`
- Balance between return and risk (λ = risk aversion)

### Risk Parity
- Equal risk contribution from each asset
- `w_i × (Σw)_i = σ_p² / N` for all i
- Diversification-focused

---

## Performance Expectations

Based on renewable energy portfolio characteristics:

- **Total Return**: 2-6% (depends on market conditions)
- **Sharpe Ratio**: 0.5-1.5 (moderate risk-adjusted returns)
- **Max Drawdown**: -5% to -15% (typical for energy portfolios)
- **Volatility**: 10-20% annualized

These are realistic for a traditional portfolio without AI optimization.

---

## Comparison: Old vs Improved

| Feature | Old Version | Improved Version |
|---------|-------------|------------------|
| Returns | Random noise | Actual renewable revenue |
| Realism | Artificial constraints | Market-based dynamics |
| Methods | 5 methods (but broken) | 5 methods (working) |
| Performance | Unrealistic | Realistic baseline |
| Integration | Partial | Full compatibility |

---

## Troubleshooting

### "Missing required columns"
- Ensure data has: `wind`, `solar`, `hydro`, `price`, `timestamp`

### "Not enough data for optimization"
- Need at least 30 timesteps before first rebalance
- Reduce `lookback_window` if dataset is small

### "SLSQP did not converge"
- Normal warning - optimizer falls back to equal weights
- Usually happens with insufficient data or high volatility

### Poor performance
- Try different optimization methods
- Adjust `risk_aversion_lambda` (default: 5.0)
- Increase `lookback_window` for more stable estimates

---

## Next Steps

1. **Test the improved baseline**: `python test_improved_baseline1.py`
2. **Run full evaluation**: Use the command above with 10,000 timesteps
3. **Compare with AI models**: Use `python run_normal_vs_baselines.py`
4. **Analyze results**: Check the output JSON files and plots

---

## References

- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
- Maillard, S. et al. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"

---

## Questions?

Check the code comments in `traditional_portfolio_optimizer_v2.py` for detailed implementation notes.


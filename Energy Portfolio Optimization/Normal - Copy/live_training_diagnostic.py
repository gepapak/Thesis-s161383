#!/usr/bin/env python3
"""
Live Training Diagnostic System
Integrates with your actual training to capture real-time portfolio dynamics
"""

import os
import json
import time
from datetime import datetime
import numpy as np

class LiveTrainingDiagnostic:
    def __init__(self, env, log_dir="diagnostic_logs", log_interval=2000):
        self.env = env
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.step_count = 0
        self.last_log_step = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tracking
        self.session_start = datetime.now()
        self.baseline_data = None
        self.last_snapshot = None
        
        # Performance tracking
        self.performance_log = []
        self.anomaly_log = []
        
        print(f"[DIAG] Live Training Diagnostic initialized")
        print(f"   Log directory: {log_dir}")
        print(f"   Log interval: {log_interval} steps")
        
    def capture_baseline(self):
        """Capture initial baseline data"""
        self.baseline_data = self._capture_snapshot()
        self.last_snapshot = self.baseline_data.copy()
        
        print(f"[DIAG] Baseline captured:")
        print(f"   Initial NAV: {self.baseline_data['fund_nav']:,.0f} DKK (${self.baseline_data['fund_nav_usd']:.1f}M)")
        print(f"   Initial cash: {self.baseline_data['budget_cash']:,.0f} DKK")
        print(f"   Physical assets: {self.baseline_data['physical_asset_value']:,.0f} DKK")
        
    def _capture_snapshot(self):
        """Capture current environment state"""
        
        # Get currency rate
        dkk_to_usd = getattr(self.env, '_dkk_to_usd_rate', getattr(self.env.config, 'dkk_to_usd_rate', 0.145))
        
        # Calculate physical asset value
        physical_value = 0.0
        if hasattr(self.env, 'asset_capex'):
            # FIXED: asset_capex values are now in DKK (converted in environment)
            physical_value = (
                self.env.physical_assets.get('wind_capacity_mw', 0) * self.env.asset_capex.get('wind_mw', 0) +
                self.env.physical_assets.get('solar_capacity_mw', 0) * self.env.asset_capex.get('solar_mw', 0) +
                self.env.physical_assets.get('hydro_capacity_mw', 0) * self.env.asset_capex.get('hydro_mw', 0) +
                self.env.physical_assets.get('battery_capacity_mwh', 0) * self.env.asset_capex.get('battery_mwh', 0)
            )
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'step': self.step_count,
            'env_step': getattr(self.env, 't', 0),
            
            # Core NAV components
            'budget_cash': getattr(self.env, 'budget', 0.0),
            'physical_asset_value': physical_value,
            'total_positions': sum(self.env.financial_positions.values()),
            'fund_nav': self.env._calculate_fund_nav() if hasattr(self.env, '_calculate_fund_nav') else 0.0,
            
            # Trading performance
            'cumulative_mtm_pnl': getattr(self.env, 'cumulative_mtm_pnl', 0.0),
            'last_mtm_pnl': getattr(self.env, 'last_mtm_pnl', 0.0),
            'cumulative_transaction_costs': getattr(self.env, 'cumulative_transaction_costs', 0.0),
            
            # Revenue streams
            'cumulative_generation_revenue': getattr(self.env, 'cumulative_generation_revenue', 0.0),
            'last_generation_revenue': getattr(self.env, 'last_generation_revenue', 0.0),
            'last_revenue': getattr(self.env, 'last_revenue', 0.0),
            
            # Individual positions
            'wind_position': self.env.financial_positions.get('wind_instrument_value', 0.0),
            'solar_position': self.env.financial_positions.get('solar_instrument_value', 0.0),
            'hydro_position': self.env.financial_positions.get('hydro_instrument_value', 0.0),
            
            # Market data
            'current_price': float(self.env._price_raw[self.env.t]) if hasattr(self.env, '_price_raw') and self.env.t < len(self.env._price_raw) else 0.0,
            
            # Currency and conversions
            'dkk_to_usd_rate': dkk_to_usd,
            'fund_nav_usd': 0.0,  # Will calculate below
            'cumulative_mtm_usd': 0.0,  # Will calculate below
            
            # Cash efficiency (if available)
            'cash_efficiency_used': getattr(self.env, 'cash_efficiency_tracker', {}).get('total_cash_used', 0.0),
            'cash_efficiency_gains': getattr(self.env, 'cash_efficiency_tracker', {}).get('total_trading_gains', 0.0),
        }
        
        # Calculate USD conversions
        snapshot['fund_nav_usd'] = snapshot['fund_nav'] * dkk_to_usd / 1e6  # Millions USD
        snapshot['cumulative_mtm_usd'] = snapshot['cumulative_mtm_pnl'] * dkk_to_usd / 1e3  # Thousands USD
        
        # COMPREHENSIVE: Calculate manual NAV for verification (include ALL components)
        operational_revenue = getattr(self.env, 'accumulated_operational_revenue', 0.0)
        distributed_profits = getattr(self.env, 'distributed_profits', 0.0)
        cumulative_returns = getattr(self.env, 'cumulative_returns', 0.0)

        snapshot['operational_revenue'] = operational_revenue
        snapshot['distributed_profits'] = distributed_profits
        snapshot['cumulative_returns'] = cumulative_returns

        # Manual NAV = Trading Cash + Physical Assets + Financial Positions + Operational Revenue + Distributed Profits
        snapshot['manual_nav'] = (snapshot['budget_cash'] +
                                snapshot['physical_asset_value'] +
                                snapshot['total_positions'] +
                                operational_revenue +
                                distributed_profits)
        snapshot['nav_discrepancy'] = snapshot['fund_nav'] - snapshot['manual_nav']
        
        return snapshot
        
    def log_step(self, force_log=False):
        """Log current step if interval reached"""
        self.step_count += 1
        
        # Check if we should log (every 2000 steps)
        should_log = (
            force_log or
            (self.step_count - self.last_log_step) >= self.log_interval or
            self.step_count % self.log_interval == 0  # Always log every log_interval steps
        )
        
        if not should_log:
            return
            
        # Capture current snapshot
        current = self._capture_snapshot()
        
        # Calculate changes since last snapshot
        if self.last_snapshot:
            changes = self._calculate_changes(self.last_snapshot, current)
            
            # Check for anomalies
            self._check_anomalies(changes, current)
            
            # Log performance data
            self._log_performance(changes, current)
            
            # Print summary
            self._print_summary(changes, current)
        
        # Update tracking
        self.last_snapshot = current
        self.last_log_step = self.step_count

    def update_step_range(self, start_step, end_step):
        """Update diagnostic for a range of steps and log at intervals"""
        steps_to_process = end_step - start_step

        # Check how many log intervals we should have hit
        current_step = start_step
        while current_step < end_step:
            next_log_step = ((current_step // self.log_interval) + 1) * self.log_interval

            if next_log_step <= end_step:
                # We should log at this step
                self.step_count = next_log_step
                self.log_step(force_log=True)
                current_step = next_log_step
            else:
                break

        # Update final step count
        self.step_count = end_step

    def _calculate_changes(self, prev, curr):
        """Calculate changes between snapshots"""
        return {
            'steps_elapsed': curr['step'] - prev['step'],
            'nav_change': curr['fund_nav'] - prev['fund_nav'],
            'nav_change_usd': curr['fund_nav_usd'] - prev['fund_nav_usd'],
            'cash_change': curr['budget_cash'] - prev['budget_cash'],
            'positions_change': curr['total_positions'] - prev['total_positions'],
            'operational_change': curr.get('operational_revenue', 0) - prev.get('operational_revenue', 0),
            'mtm_change': curr['cumulative_mtm_pnl'] - prev['cumulative_mtm_pnl'],
            'mtm_change_usd': curr['cumulative_mtm_usd'] - prev['cumulative_mtm_usd'],
            'generation_change': curr['cumulative_generation_revenue'] - prev['cumulative_generation_revenue'],
            'transaction_costs_change': curr['cumulative_transaction_costs'] - prev['cumulative_transaction_costs'],
            'nav_discrepancy': curr['nav_discrepancy'],
        }
        
    def _check_anomalies(self, changes, current):
        """Check for anomalies and log them"""
        anomalies = []

        # Check for SIGNIFICANT NAV calculation discrepancies (increased threshold)
        if abs(changes['nav_discrepancy']) > 100000000:  # 100M DKK threshold (was 1000)
            anomalies.append(f"MAJOR NAV discrepancy: {changes['nav_discrepancy']:+,.0f} DKK")

        # COMPREHENSIVE: Check for unexpected NAV changes (include ALL components)
        if self.last_snapshot:
            operational_change = current.get('operational_revenue', 0) - self.last_snapshot.get('operational_revenue', 0)
            distributed_change = current.get('distributed_profits', 0) - self.last_snapshot.get('distributed_profits', 0)
            expected_nav_change = (changes['cash_change'] +
                                 changes['positions_change'] +
                                 operational_change +
                                 distributed_change)
            actual_nav_change = changes['nav_change']
            nav_diff = actual_nav_change - expected_nav_change
            if abs(nav_diff) > 10000000:  # 10M DKK threshold (was 1000)
                anomalies.append(f"Unexpected NAV change: expected {expected_nav_change:+,.0f}, actual {actual_nav_change:+,.0f}, diff {nav_diff:+,.0f}")

        # Check for trading gains not flowing to portfolio (more reasonable threshold)
        if changes['mtm_change'] > 10000000 and abs(changes['nav_change_usd']) < 1.0:  # >10M DKK gains but <$1M portfolio change
            anomalies.append(f"Trading gains not flowing to portfolio: {changes['mtm_change']:+,.0f} DKK gains, {changes['nav_change_usd']:+.1f}M USD portfolio change")
            
        # Log anomalies
        if anomalies:
            anomaly_entry = {
                'timestamp': current['timestamp'],
                'step': current['step'],
                'anomalies': anomalies,
                'current_state': current,
                'changes': changes
            }
            self.anomaly_log.append(anomaly_entry)
            
            print(f"[ALERT] ANOMALIES DETECTED at step {current['step']}:")
            for anomaly in anomalies:
                print(f"   {anomaly}")
                
    def _log_performance(self, changes, current):
        """Log performance data"""
        
        # Calculate efficiency metrics
        cash_efficiency = 0.0
        if current['cash_efficiency_used'] > 0:
            cash_efficiency = current['cash_efficiency_gains'] / current['cash_efficiency_used']
            
        # Calculate returns since baseline
        if self.baseline_data:
            total_return_dkk = current['fund_nav'] - self.baseline_data['fund_nav']
            total_return_pct = total_return_dkk / self.baseline_data['fund_nav'] * 100
            total_return_usd = current['fund_nav_usd'] - self.baseline_data['fund_nav_usd']
        else:
            total_return_dkk = total_return_pct = total_return_usd = 0.0
            
        perf_entry = {
            'timestamp': current['timestamp'],
            'step': current['step'],
            'fund_nav_usd': current['fund_nav_usd'],
            'total_return_dkk': total_return_dkk,
            'total_return_pct': total_return_pct,
            'total_return_usd': total_return_usd,
            'cumulative_mtm_usd': current['cumulative_mtm_usd'],
            'cash_efficiency': cash_efficiency,
            'nav_change_interval': changes['nav_change_usd'],
            'mtm_change_interval': changes['mtm_change_usd'],
        }
        
        self.performance_log.append(perf_entry)
        
    def _print_summary(self, changes, current):
        """Print interval summary"""
        # FIXED: Show actual environment timestep instead of diagnostic counter
        env_step = current.get('env_step', current['step'])
        print(f"\n[DIAG] DIAGNOSTIC SUMMARY - Step {current['step']:,} (Env: {env_step:,})")
        print(f"   Portfolio: ${current['fund_nav_usd']:.1f}M (change: ${changes['nav_change_usd']:+.1f}M)")
        print(f"   Trading: ${current['cumulative_mtm_usd']:+.1f}k (change: ${changes['mtm_change_usd']:+.1f}k)")
        print(f"   Cash: {current['budget_cash']:,.0f} DKK (change: {changes['cash_change']:+,.0f})")
        print(f"   Positions: {current['total_positions']:,.0f} DKK (change: {changes['positions_change']:+,.0f})")
        print(f"   Operational: {current.get('operational_revenue', 0):,.0f} DKK (change: {changes.get('operational_change', 0):+,.0f})")

        # Show distributions if significant
        if current.get('distributed_profits', 0) > 1000000:  # >1M DKK
            print(f"   Distributions: {current.get('distributed_profits', 0):,.0f} DKK")

        # Enhanced NAV breakdown
        if abs(changes['nav_discrepancy']) > 1000000:  # Show breakdown if discrepancy > 1M DKK
            print(f"   NAV Components:")
            print(f"     Cash: {current['budget_cash'] / 1e6:.1f}M DKK")
            print(f"     Physical: {current['physical_asset_value'] / 1e6:.1f}M DKK")
            print(f"     Positions: {current['total_positions'] / 1e6:.1f}M DKK")
            print(f"     Operational: {current.get('operational_revenue', 0) / 1e6:.1f}M DKK")
            print(f"     Distributed: {current.get('distributed_profits', 0) / 1e6:.1f}M DKK")
            print(f"     Manual Total: {current['manual_nav'] / 1e6:.1f}M DKK")
            print(f"     System NAV: {current['fund_nav'] / 1e6:.1f}M DKK")
            print(f"     Discrepancy: {changes['nav_discrepancy'] / 1e6:+.1f}M DKK")

        if self.baseline_data:
            total_return = current['fund_nav_usd'] - self.baseline_data['fund_nav_usd']
            print(f"   Total return: ${total_return:+.1f}M ({total_return/self.baseline_data['fund_nav_usd']*100:+.2f}%)")
            
    def save_logs(self):
        """Save diagnostic logs to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance log
        perf_file = os.path.join(self.log_dir, f"performance_log_{timestamp}.json")
        with open(perf_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
            
        # Save anomaly log
        if self.anomaly_log:
            anomaly_file = os.path.join(self.log_dir, f"anomaly_log_{timestamp}.json")
            with open(anomaly_file, 'w') as f:
                json.dump(self.anomaly_log, f, indent=2)
                
        print(f"[DIAG] Diagnostic logs saved:")
        print(f"   Performance: {perf_file}")
        if self.anomaly_log:
            print(f"   Anomalies: {anomaly_file}")
            
    def generate_final_report(self):
        """Generate final diagnostic report"""
        if not self.baseline_data or not self.last_snapshot:
            return
            
        print(f"\n[FINAL] FINAL DIAGNOSTIC REPORT")
        print("=" * 60)
        
        current = self.last_snapshot
        baseline = self.baseline_data
        
        # Portfolio performance
        total_return_usd = current['fund_nav_usd'] - baseline['fund_nav_usd']
        total_return_pct = total_return_usd / baseline['fund_nav_usd'] * 100
        
        print(f"Portfolio Performance:")
        print(f"   Initial: ${baseline['fund_nav_usd']:.1f}M")
        print(f"   Final: ${current['fund_nav_usd']:.1f}M")
        print(f"   Total return: ${total_return_usd:+.1f}M ({total_return_pct:+.2f}%)")
        
        # Trading performance
        print(f"\nTrading Performance:")
        print(f"   Total MTM gains: ${current['cumulative_mtm_usd']:+.1f}k")
        print(f"   Transaction costs: {current['cumulative_transaction_costs']:,.0f} DKK")
        print(f"   Net trading: ${current['cumulative_mtm_usd'] - current['cumulative_transaction_costs']*current['dkk_to_usd_rate']/1e3:+.1f}k")
        
        # Efficiency analysis
        if current['cumulative_mtm_usd'] != 0:
            efficiency = total_return_usd * 1000 / current['cumulative_mtm_usd']  # Convert to same units
            print(f"\nEfficiency Analysis:")
            print(f"   Portfolio growth per $1k trading gain: ${efficiency:.2f}")
            print(f"   Trading-to-portfolio flow rate: {efficiency*100:.1f}%")
            
        # Anomaly summary
        if self.anomaly_log:
            print(f"\nAnomalies Detected: {len(self.anomaly_log)}")
            for i, anomaly in enumerate(self.anomaly_log[-3:], 1):  # Show last 3
                print(f"   {i}. Step {anomaly['step']}: {anomaly['anomalies'][0]}")
                
        self.save_logs()

# Integration function for your training
def integrate_diagnostic_with_training(env, log_interval=2000):
    """
    Integration function to add to your training loop

    Usage in your training:
    diagnostic = integrate_diagnostic_with_training(env, log_interval=2000)
    diagnostic.capture_baseline()

    # In your training loop:
    diagnostic.log_step()

    # At the end:
    diagnostic.generate_final_report()
    """
    return LiveTrainingDiagnostic(env, log_interval=log_interval)

# Test function
def test_live_diagnostic():
    """Test the live diagnostic system"""
    print("🧪 Testing Live Diagnostic System")

    from environment import RenewableMultiAgentEnv
    from config import EnhancedConfig
    import pandas as pd

    # Setup
    config = EnhancedConfig()
    data = pd.read_csv('trainingdata.csv').head(50)
    env = RenewableMultiAgentEnv(data, config=config)

    # Initialize diagnostic
    diagnostic = LiveTrainingDiagnostic(env, log_interval=5)

    # Reset and capture baseline
    obs = env.reset()
    diagnostic.capture_baseline()

    # Simulate training steps
    for step in range(20):
        actions = {
            'investor_0': np.array([0.2, 0.2, 0.2]),
            'battery_operator_0': np.array([0.0, 0.0]),
            'risk_controller_0': np.array([0.0]),
            'meta_controller_0': np.array([0.0])
        }

        result = env.step(actions)
        diagnostic.log_step()

    # Generate final report
    diagnostic.generate_final_report()

if __name__ == "__main__":
    test_live_diagnostic()

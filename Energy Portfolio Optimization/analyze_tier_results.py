#!/usr/bin/env python3
"""
Deep analysis of Tier 1 vs Tier 2 results to understand why Tier 2 underperformed.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_episode_portfolio(episode_file):
    """Extract key metrics from a portfolio CSV file."""
    try:
        df = pd.read_csv(episode_file)
        
        # Find NAV column (might be named differently)
        nav_cols = [col for col in df.columns if 'nav' in col.lower() or 'portfolio' in col.lower()]
        if not nav_cols:
            # Try to infer from numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                nav_col = numeric_cols[0]  # Assume first numeric column
            else:
                return None
        else:
            nav_col = nav_cols[0]
        
        nav_values = df[nav_col].values
        
        return {
            'final_nav': nav_values[-1] if len(nav_values) > 0 else None,
            'max_nav': np.max(nav_values) if len(nav_values) > 0 else None,
            'min_nav': np.min(nav_values) if len(nav_values) > 0 else None,
            'mean_nav': np.mean(nav_values) if len(nav_values) > 0 else None,
            'std_nav': np.std(nav_values) if len(nav_values) > 0 else None,
            'final_return': (nav_values[-1] / nav_values[0] - 1) * 100 if len(nav_values) > 1 else None,
            'max_drawdown': ((nav_values / np.maximum.accumulate(nav_values)) - 1).min() * 100 if len(nav_values) > 0 else None,
            'num_steps': len(nav_values)
        }
    except Exception as e:
        print(f"Error analyzing {episode_file}: {e}")
        return None

def analyze_episode_rewards(episode_file):
    """Extract reward statistics from a rewards CSV file."""
    try:
        df = pd.read_csv(episode_file)
        
        # Find reward columns
        reward_cols = [col for col in df.columns if 'reward' in col.lower() or 'total' in col.lower()]
        if not reward_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                reward_col = numeric_cols[0]
            else:
                return None
        else:
            reward_col = reward_cols[0]
        
        rewards = df[reward_col].values
        
        return {
            'total_reward': np.sum(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'positive_reward_ratio': np.sum(rewards > 0) / len(rewards) if len(rewards) > 0 else 0
        }
    except Exception as e:
        print(f"Error analyzing rewards {episode_file}: {e}")
        return None

def analyze_all_episodes(tier_dir, tier_name):
    """Analyze all episodes for a tier."""
    logs_dir = Path(tier_dir) / "logs"
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return None
    
    portfolio_files = sorted(logs_dir.glob(f"{tier_name}_portfolio_*.csv"))
    
    if not portfolio_files:
        print(f"No portfolio files found in {logs_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Analyzing {tier_name.upper()} - {len(portfolio_files)} episodes")
    print(f"{'='*80}")
    
    episode_stats = []
    
    for pf_file in portfolio_files:
        ep_num = int(pf_file.stem.split('_')[-1].replace('ep', ''))
        
        # Get portfolio stats
        portfolio_stats = analyze_episode_portfolio(pf_file)
        
        # Get rewards stats
        reward_file = logs_dir / f"{tier_name}_rewards_ep{ep_num}.csv"
        reward_stats = analyze_episode_rewards(reward_file) if reward_file.exists() else None
        
        if portfolio_stats:
            stats = {'episode': ep_num, **portfolio_stats}
            if reward_stats:
                stats.update({f'reward_{k}': v for k, v in reward_stats.items()})
            episode_stats.append(stats)
    
    if not episode_stats:
        return None
    
    df_stats = pd.DataFrame(episode_stats)
    
    # Print summary statistics
    print(f"\nPortfolio Performance Summary:")
    print(f"  Final NAV: Mean={df_stats['final_nav'].mean():.2f}, Std={df_stats['final_nav'].std():.2f}")
    print(f"  Max NAV: Mean={df_stats['max_nav'].mean():.2f}, Std={df_stats['max_nav'].std():.2f}")
    print(f"  Final Return: Mean={df_stats['final_return'].mean():.2f}%, Std={df_stats['final_return'].std():.2f}%")
    print(f"  Max Drawdown: Mean={df_stats['max_drawdown'].mean():.2f}%, Std={df_stats['max_drawdown'].std():.2f}%")
    
    if 'reward_total_reward' in df_stats.columns:
        print(f"\nReward Statistics:")
        print(f"  Total Reward: Mean={df_stats['reward_total_reward'].mean():.2f}, Std={df_stats['reward_total_reward'].std():.2f}")
        print(f"  Mean Reward: Mean={df_stats['reward_mean_reward'].mean():.6f}, Std={df_stats['reward_mean_reward'].std():.6f}")
        print(f"  Positive Reward Ratio: Mean={df_stats['reward_positive_reward_ratio'].mean():.2%}")
    
    # Print last 5 episodes
    print(f"\nLast 5 Episodes:")
    print(df_stats[['episode', 'final_nav', 'final_return', 'max_drawdown']].tail(5).to_string(index=False))
    
    return df_stats

def compare_tiers():
    """Compare Tier 1 vs Tier 2 performance."""
    
    tier1_dir = "tier1gnn_seed789"
    tier2_dir = "tier2gnn_seed789"
    
    print("="*80)
    print("TIER COMPARISON ANALYSIS")
    print("="*80)
    
    tier1_stats = analyze_all_episodes(tier1_dir, "tier1")
    tier2_stats = analyze_all_episodes(tier2_dir, "tier2")
    
    if tier1_stats is None or tier2_stats is None:
        print("\nERROR: Could not analyze one or both tiers")
        return
    
    print(f"\n{'='*80}")
    print("TIER 1 vs TIER 2 COMPARISON")
    print(f"{'='*80}")
    
    # Compare key metrics
    metrics = ['final_nav', 'max_nav', 'final_return', 'max_drawdown']
    
    print("\nMetric Comparison (Tier 1 vs Tier 2):")
    print(f"{'Metric':<20} {'Tier 1 Mean':<15} {'Tier 2 Mean':<15} {'Difference':<15} {'% Change':<15}")
    print("-" * 80)
    
    for metric in metrics:
        t1_mean = tier1_stats[metric].mean()
        t2_mean = tier2_stats[metric].mean()
        diff = t2_mean - t1_mean
        pct_change = (diff / abs(t1_mean)) * 100 if abs(t1_mean) > 1e-6 else 0
        
        print(f"{metric:<20} {t1_mean:>14.2f} {t2_mean:>14.2f} {diff:>14.2f} {pct_change:>14.2f}%")
    
    # Compare rewards if available
    if 'reward_total_reward' in tier1_stats.columns and 'reward_total_reward' in tier2_stats.columns:
        print("\nReward Comparison:")
        t1_reward = tier1_stats['reward_total_reward'].mean()
        t2_reward = tier2_stats['reward_total_reward'].mean()
        reward_diff = t2_reward - t1_reward
        reward_pct = (reward_diff / abs(t1_reward)) * 100 if abs(t1_reward) > 1e-6 else 0
        
        print(f"  Total Reward: Tier 1={t1_reward:.2f}, Tier 2={t2_reward:.2f}, Diff={reward_diff:.2f} ({reward_pct:.2f}%)")
        
        t1_mean_reward = tier1_stats['reward_mean_reward'].mean()
        t2_mean_reward = tier2_stats['reward_mean_reward'].mean()
        print(f"  Mean Reward: Tier 1={t1_mean_reward:.6f}, Tier 2={t2_mean_reward:.6f}")
    
    # Episode-by-episode comparison
    print("\n" + "="*80)
    print("EPISODE-BY-EPISODE COMPARISON (Last 10 Episodes)")
    print("="*80)
    
    # Merge on episode number
    comparison = pd.merge(
        tier1_stats[['episode', 'final_nav', 'final_return', 'max_drawdown']].rename(columns={
            'final_nav': 'tier1_nav',
            'final_return': 'tier1_return',
            'max_drawdown': 'tier1_dd'
        }),
        tier2_stats[['episode', 'final_nav', 'final_return', 'max_drawdown']].rename(columns={
            'final_nav': 'tier2_nav',
            'final_return': 'tier2_return',
            'max_drawdown': 'tier2_dd'
        }),
        on='episode',
        how='inner'
    )
    
    comparison['nav_diff'] = comparison['tier2_nav'] - comparison['tier1_nav']
    comparison['nav_pct_diff'] = (comparison['nav_diff'] / comparison['tier1_nav']) * 100
    
    print(comparison.tail(10).to_string(index=False))
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    nav_wins_tier2 = (comparison['nav_diff'] > 0).sum()
    nav_wins_tier1 = (comparison['nav_diff'] < 0).sum()
    nav_ties = (comparison['nav_diff'] == 0).sum()
    
    print(f"\nEpisodes where Tier 2 outperformed Tier 1: {nav_wins_tier2}/{len(comparison)} ({nav_wins_tier2/len(comparison)*100:.1f}%)")
    print(f"Episodes where Tier 1 outperformed Tier 2: {nav_wins_tier1}/{len(comparison)} ({nav_wins_tier1/len(comparison)*100:.1f}%)")
    print(f"Episodes tied: {nav_ties}/{len(comparison)} ({nav_ties/len(comparison)*100:.1f}%)")
    
    avg_nav_diff = comparison['nav_diff'].mean()
    avg_pct_diff = comparison['nav_pct_diff'].mean()
    
    print(f"\nAverage NAV difference: {avg_nav_diff:.2f} ({avg_pct_diff:.2f}%)")
    
    if avg_nav_diff < 0:
        print("\n⚠️  TIER 2 IS UNDERPERFORMING TIER 1")
        print(f"   Average underperformance: {abs(avg_nav_diff):.2f} ({abs(avg_pct_diff):.2f}%)")
    else:
        print("\n✅ TIER 2 IS OUTPERFORMING TIER 1")
        print(f"   Average outperformance: {avg_nav_diff:.2f} ({avg_pct_diff:.2f}%)")
    
    return tier1_stats, tier2_stats, comparison

if __name__ == "__main__":
    tier1_stats, tier2_stats, comparison = compare_tiers()
    
    # Save results
    if comparison is not None:
        comparison.to_csv("tier_comparison_analysis.csv", index=False)
        print("\n✅ Saved detailed comparison to tier_comparison_analysis.csv")

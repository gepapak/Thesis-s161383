#!/usr/bin/env python3
"""
Offline Interactive Dashboard for CSV Analysis
Generates a standalone HTML file with no external dependencies (pure JS/CSS/HTML5)
Visualizes portfolio performance, forecasts, rewards, positions, and MAPE metrics

This module provides:
1. OfflineDashboardGenerator: Core dashboard generation class
2. main(): Convenient entry point to generate dashboards for multiple tiers
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import base64
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)


class OfflineDashboardGenerator:
    """Generate offline HTML dashboard from tier training CSVs"""
    
    def __init__(self, tier_dir: str):
        self.tier_dir = tier_dir
        self.tier_name = os.path.basename(tier_dir)
        self.logs_dir = os.path.join(tier_dir, 'logs')
        
    def load_episode_data(self, episode_num: int) -> Dict[str, pd.DataFrame]:
        """Load all CSVs for a given episode"""
        data = {}
        
        # Load category-specific CSVs
        for category in ['portfolio', 'forecast', 'rewards', 'positions', 'debug']:
            csv_file = os.path.join(self.logs_dir, f"{self.tier_name}_debug_ep{episode_num}.csv")
            if category != 'debug':
                csv_file = os.path.join(self.logs_dir, f"{self.tier_name}_{category}_ep{episode_num}.csv")
            
            if os.path.exists(csv_file):
                try:
                    data[category] = pd.read_csv(csv_file)
                except Exception as e:
                    logger.warning(f"Could not load {category} CSV: {e}")
        
        return data
    
    def get_available_episodes(self) -> List[int]:
        """Find all available episodes"""
        if not os.path.exists(self.logs_dir):
            return []
        
        episodes = set()
        for f in os.listdir(self.logs_dir):
            if f.endswith('.csv') and 'ep' in f:
                try:
                    ep_num = int(f.split('ep')[-1].replace('.csv', ''))
                    episodes.add(ep_num)
                except:
                    pass
        return sorted(list(episodes))
    
    def generate_html(self, output_file: str = None):
        """Generate standalone HTML dashboard"""
        if output_file is None:
            output_file = os.path.join(self.tier_dir, f'{self.tier_name}_dashboard.html')
        
        episodes = self.get_available_episodes()
        
        if not episodes:
            logger.warning(f"No episodes found in {self.logs_dir}")
            return
        
        # Load sample episode data for structure
        sample_ep = episodes[0]
        sample_data = self.load_episode_data(sample_ep)
        
        # Create HTML
        html = self._create_html_template(episodes, sample_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Dashboard created: {output_file}")
        logger.info(f"Open in browser to view: file://{os.path.abspath(output_file)}")
    
    def _create_html_template(self, episodes: List[int], sample_data: Dict) -> str:
        """Create the HTML dashboard template with embedded data"""
        
        # Prepare episode data in JSON format
        all_episodes_data = {}
        for ep in episodes[:10]:  # Load first 10 episodes to avoid huge file
            try:
                ep_data = self.load_episode_data(ep)
                if 'debug' in ep_data and len(ep_data['debug']) > 0:
                    debug_df = ep_data['debug']
                    all_episodes_data[f"episode_{ep}"] = {
                        'portfolio_value': debug_df['portfolio_value_usd_millions'].tolist() if 'portfolio_value_usd_millions' in debug_df.columns else [],
                        'timesteps': debug_df['timestep'].tolist() if 'timestep' in debug_df.columns else [],
                        'base_reward': debug_df['base_reward'].tolist() if 'base_reward' in debug_df.columns else [],
                        'investor_reward': debug_df['investor_reward'].tolist() if 'investor_reward' in debug_df.columns else [],
                        'battery_reward': debug_df['battery_reward'].tolist() if 'battery_reward' in debug_df.columns else [],
                        'forecast_confidence': debug_df['forecast_confidence'].tolist() if 'forecast_confidence' in debug_df.columns else [],
                        'mape_short': debug_df['mape_short'].tolist() if 'mape_short' in debug_df.columns else [],
                        'mape_medium': debug_df['mape_medium'].tolist() if 'mape_medium' in debug_df.columns else [],
                        'mape_long': debug_df['mape_long'].tolist() if 'mape_long' in debug_df.columns else [],
                        'position_signed': debug_df['position_signed'].tolist() if 'position_signed' in debug_df.columns else [],
                        'price_current': debug_df['price_current'].tolist() if 'price_current' in debug_df.columns else [],
                    }
            except Exception as e:
                logger.warning(f"Could not process episode {ep}: {e}")
        
        episodes_json = json.dumps(all_episodes_data)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.tier_name} Trading Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            border-left: 4px solid #3b82f6;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
            color: #60a5fa;
        }}
        
        .header p {{
            color: #94a3b8;
            font-size: 14px;
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
            background: rgba(30, 41, 59, 0.8);
            padding: 20px;
            border-radius: 12px;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .control-group label {{
            color: #cbd5e1;
            font-weight: 500;
            font-size: 14px;
        }}
        
        .control-group select, .control-group input {{
            padding: 8px 12px;
            border: 1px solid #475569;
            border-radius: 6px;
            background: #1e293b;
            color: #e2e8f0;
            font-size: 14px;
            cursor: pointer;
        }}
        
        .control-group select:hover, .control-group input:hover {{
            border-color: #3b82f6;
        }}
        
        .control-group select:focus, .control-group input:focus {{
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            border-color: #3b82f6;
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.1);
        }}
        
        .card h3 {{
            color: #60a5fa;
            margin-bottom: 15px;
            font-size: 16px;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 10px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        
        .stat {{
            background: rgba(15, 23, 42, 0.8);
            padding: 12px;
            border-radius: 8px;
            border-left: 3px solid #3b82f6;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #60a5fa;
            margin-top: 5px;
        }}
        
        .stat.positive .stat-value {{
            color: #4ade80;
        }}
        
        .stat.negative .stat-value {{
            color: #f87171;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #334155;
            color: #64748b;
            font-size: 12px;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 20px;
            }}
            
            .controls {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {self.tier_name.upper()} Trading Analysis Dashboard</h1>
            <p>Real-time performance metrics from checkpoint CSVs</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="episodeSelect">📊 Episode:</label>
                <select id="episodeSelect" onchange="updateDashboard()">
                    {self._generate_episode_options(episodes)}
                </select>
            </div>
            <div class="control-group">
                <label for="refreshBtn">🔄</label>
                <button id="refreshBtn" onclick="location.reload()" style="padding: 8px 16px; background: #3b82f6; border: none; border-radius: 6px; color: white; cursor: pointer;">Refresh</button>
            </div>
        </div>
        
        <div class="grid">
            <!-- Portfolio Performance -->
            <div class="card">
                <h3>💰 Portfolio Value</h3>
                <div class="chart-container">
                    <canvas id="portfolioChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat" id="stat-portfolio-start">
                        <div class="stat-label">Start</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-portfolio-end">
                        <div class="stat-label">End</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-portfolio-return">
                        <div class="stat-label">Return</div>
                        <div class="stat-value">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Rewards Composition -->
            <div class="card">
                <h3>🎯 Rewards Breakdown</h3>
                <div class="chart-container">
                    <canvas id="rewardsChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat" id="stat-base-reward">
                        <div class="stat-label">Avg Base</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-investor-reward">
                        <div class="stat-label">Avg Investor</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-battery-reward">
                        <div class="stat-label">Avg Battery</div>
                        <div class="stat-value">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Forecast Accuracy (MAPE) -->
            <div class="card">
                <h3>🎲 Forecast MAPE by Horizon</h3>
                <div class="chart-container">
                    <canvas id="mapeChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat" id="stat-mape-short">
                        <div class="stat-label">Short</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-mape-medium">
                        <div class="stat-label">Medium</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-mape-long">
                        <div class="stat-label">Long</div>
                        <div class="stat-value">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Forecast Confidence -->
            <div class="card">
                <h3>📈 Forecast Trust Over Time</h3>
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat" id="stat-confidence-avg">
                        <div class="stat-label">Avg Trust</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-confidence-min">
                        <div class="stat-label">Min Trust</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-confidence-max">
                        <div class="stat-label">Max Trust</div>
                        <div class="stat-value">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Position Over Time -->
            <div class="card">
                <h3>📍 Position & Price</h3>
                <div class="chart-container">
                    <canvas id="positionChart"></canvas>
                </div>
                <div class="stats">
                    <div class="stat" id="stat-position-avg">
                        <div class="stat-label">Avg Position</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-price-start">
                        <div class="stat-label">Price Start</div>
                        <div class="stat-value">-</div>
                    </div>
                    <div class="stat" id="stat-price-end">
                        <div class="stat-label">Price End</div>
                        <div class="stat-value">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Cumulative Reward -->
            <div class="card full-width">
                <h3>📊 Cumulative Reward Over Episode</h3>
                <div class="chart-container" style="height: 250px;">
                    <canvas id="cumulativeChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Dashboard generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Data source: {self.tier_name}/logs/*.csv</p>
        </div>
    </div>
    
    <script>
        // Embedded episode data
        const allEpisodesData = {episodes_json};
        
        let charts = {{}};
        
        function updateDashboard() {{
            const episode = document.getElementById('episodeSelect').value;
            const data = allEpisodesData[episode];
            
            if (!data || data.timesteps.length === 0) {{
                console.warn('No data for episode:', episode);
                return;
            }}
            
            updatePortfolioChart(data);
            updateRewardsChart(data);
            updateMapeChart(data);
            updateConfidenceChart(data);
            updatePositionChart(data);
            updateCumulativeChart(data);
            updateStats(data);
        }}
        
        function updatePortfolioChart(data) {{
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            if (charts.portfolio) charts.portfolio.destroy();
            
            charts.portfolio = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps.slice(0, Math.min(100, data.timesteps.length)),
                    datasets: [{{
                        label: 'Portfolio Value (USD M)',
                        data: data.portfolio_value.slice(0, Math.min(100, data.portfolio_value.length)),
                        borderColor: '#60a5fa',
                        backgroundColor: 'rgba(96, 165, 250, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 2,
                        pointBackgroundColor: '#3b82f6'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{legend: {{display: false}}}}
                }}
            }});
        }}
        
        function updateRewardsChart(data) {{
            const ctx = document.getElementById('rewardsChart').getContext('2d');
            if (charts.rewards) charts.rewards.destroy();
            
            charts.rewards = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps.slice(0, Math.min(100, data.timesteps.length)),
                    datasets: [
                        {{
                            label: 'Base Reward',
                            data: data.base_reward.slice(0, Math.min(100, data.base_reward.length)),
                            borderColor: '#60a5fa',
                            tension: 0.3,
                            pointRadius: 1
                        }},
                        {{
                            label: 'Investor Reward',
                            data: data.investor_reward.slice(0, Math.min(100, data.investor_reward.length)),
                            borderColor: '#4ade80',
                            tension: 0.3,
                            pointRadius: 1
                        }},
                        {{
                            label: 'Battery Reward',
                            data: data.battery_reward.slice(0, Math.min(100, data.battery_reward.length)),
                            borderColor: '#fbbf24',
                            tension: 0.3,
                            pointRadius: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{legend: {{display: true}}}}
                }}
            }});
        }}
        
        function updateMapeChart(data) {{
            const ctx = document.getElementById('mapeChart').getContext('2d');
            if (charts.mape) charts.mape.destroy();
            
            charts.mape = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps.slice(0, Math.min(100, data.timesteps.length)),
                    datasets: [
                        {{
                            label: 'Short MAPE',
                            data: data.mape_short.slice(0, Math.min(100, data.mape_short.length)),
                            borderColor: '#60a5fa',
                            tension: 0.3,
                            pointRadius: 1
                        }},
                        {{
                            label: 'Medium MAPE',
                            data: data.mape_medium.slice(0, Math.min(100, data.mape_medium.length)),
                            borderColor: '#fbbf24',
                            tension: 0.3,
                            pointRadius: 1
                        }},
                        {{
                            label: 'Long MAPE',
                            data: data.mape_long.slice(0, Math.min(100, data.mape_long.length)),
                            borderColor: '#f87171',
                            tension: 0.3,
                            pointRadius: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{legend: {{display: true}}}}
                }}
            }});
        }}
        
        function updateConfidenceChart(data) {{
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            if (charts.confidence) charts.confidence.destroy();
            
            charts.confidence = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps.slice(0, Math.min(100, data.timesteps.length)),
                    datasets: [{{
                        label: 'Forecast Trust',
                        data: data.forecast_confidence.slice(0, Math.min(100, data.forecast_confidence.length)),
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74, 222, 128, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{legend: {{display: false}}}},
                    scales: {{y: {{min: 0, max: 1}}}}
                }}
            }});
        }}
        
        function updatePositionChart(data) {{
            const ctx = document.getElementById('positionChart').getContext('2d');
            if (charts.position) charts.position.destroy();
            
            charts.position = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps.slice(0, Math.min(100, data.timesteps.length)),
                    datasets: [
                        {{
                            label: 'Position',
                            data: data.position_signed.slice(0, Math.min(100, data.position_signed.length)),
                            borderColor: '#60a5fa',
                            yAxisID: 'y',
                            tension: 0.3,
                            pointRadius: 1
                        }},
                        {{
                            label: 'Price',
                            data: data.price_current.slice(0, Math.min(100, data.price_current.length)),
                            borderColor: '#fbbf24',
                            yAxisID: 'y1',
                            tension: 0.3,
                            pointRadius: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{mode: 'index', intersect: false}},
                    scales: {{
                        y: {{position: 'left'}},
                        y1: {{position: 'right', grid: {{drawOnChartArea: false}}}}
                    }},
                    plugins: {{legend: {{display: true}}}}
                }}
            }});
        }}
        
        function updateCumulativeChart(data) {{
            const cumReward = [];
            let sum = 0;
            for (let r of data.base_reward) {{
                sum += r;
                cumReward.push(sum);
            }}
            
            const ctx = document.getElementById('cumulativeChart').getContext('2d');
            if (charts.cumulative) charts.cumulative.destroy();
            
            charts.cumulative = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timesteps,
                    datasets: [{{
                        label: 'Cumulative Reward',
                        data: cumReward,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{legend: {{display: false}}}}
                }}
            }});
        }}
        
        function updateStats(data) {{
            const stat = (id, value, format = x => x.toFixed(2)) => {{
                const el = document.getElementById(id);
                if (el) el.querySelector('.stat-value').textContent = format(value);
            }};
            
            if (data.portfolio_value.length > 0) {{
                stat('stat-portfolio-start', data.portfolio_value[0], x => x.toFixed(3));
                stat('stat-portfolio-end', data.portfolio_value[data.portfolio_value.length - 1], x => x.toFixed(3));
                const ret = (data.portfolio_value[data.portfolio_value.length - 1] / data.portfolio_value[0] - 1) * 100;
                stat('stat-portfolio-return', ret, x => x.toFixed(1) + '%');
                const cls = ret >= 0 ? 'positive' : 'negative';
                document.getElementById('stat-portfolio-return').className = 'stat ' + cls;
            }}
            
            if (data.base_reward.length > 0) {{
                stat('stat-base-reward', data.base_reward.reduce((a,b) => a+b) / data.base_reward.length);
                stat('stat-investor-reward', data.investor_reward.reduce((a,b) => a+b) / data.investor_reward.length);
                stat('stat-battery-reward', data.battery_reward.reduce((a,b) => a+b) / data.battery_reward.length);
            }}
            
            if (data.mape_short.length > 0) {{
                stat('stat-mape-short', data.mape_short.reduce((a,b) => a+b) / data.mape_short.length, x => x.toFixed(1) + '%');
                stat('stat-mape-medium', data.mape_medium.reduce((a,b) => a+b) / data.mape_medium.length, x => x.toFixed(1) + '%');
                stat('stat-mape-long', data.mape_long.reduce((a,b) => a+b) / data.mape_long.length, x => x.toFixed(1) + '%');
            }}
            
            if (data.forecast_confidence.length > 0) {{
                stat('stat-confidence-avg', data.forecast_confidence.reduce((a,b) => a+b) / data.forecast_confidence.length);
                stat('stat-confidence-min', Math.min(...data.forecast_confidence));
                stat('stat-confidence-max', Math.max(...data.forecast_confidence));
            }}
            
            if (data.position_signed.length > 0) {{
                stat('stat-position-avg', data.position_signed.reduce((a,b) => a+b) / data.position_signed.length);
                stat('stat-price-start', data.price_current[0], x => x.toFixed(0));
                stat('stat-price-end', data.price_current[data.price_current.length - 1], x => x.toFixed(0));
            }}
        }}
        
        // Initialize on load
        window.addEventListener('load', () => {{
            updateDashboard();
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_episode_options(self, episodes: List[int]) -> str:
        """Generate HTML options for episode selector"""
        options = []
        for ep in episodes[:10]:  # Show first 10
            selected = "selected" if ep == episodes[0] else ""
            options.append(f'<option value="episode_{ep}" {selected}>Episode {ep}</option>')
        return '\n                    '.join(options)


def main(tiers: List[str] = None):
    """
    Convenient entry point to generate dashboards for multiple tiers.
    
    Args:
        tiers: List of tier directory names (e.g., ['tier1', 'tier2']).
               If None, uses command-line arguments or defaults to ['tier1'].
    """
    # Handle command-line arguments if tiers not provided
    if tiers is None:
        if len(sys.argv) > 1:
            # Single tier from command line
            tier_dir = sys.argv[1]
            if not os.path.exists(tier_dir):
                logger.error(f"Directory {tier_dir} not found")
                sys.exit(1)
            generator = OfflineDashboardGenerator(tier_dir)
            generator.generate_html()
            return
        else:
            # Default: try common tier names
            tiers = ['tier1', 'tier2']
    
    # Generate dashboards for multiple tiers
    for tier in tiers:
        tier_path = tier
        
        if not os.path.exists(tier_path):
            logger.warning(f"{tier} not found, skipping...")
            continue
        
        logger.info(f"\n📊 Generating dashboard for {tier}...")
        
        try:
            gen = OfflineDashboardGenerator(tier_path)
            episodes = gen.get_available_episodes()
            
            if not episodes:
                logger.warning(f"No episodes found in {tier}/logs/")
                continue
            
            gen.generate_html()
            
            dashboard_file = os.path.join(tier_path, f'{tier}_dashboard.html')
            logger.info(f"✅ Dashboard created: {dashboard_file}")
            logger.info(f"📂 Episodes found: {len(episodes)} (ep {episodes[0]}-{episodes[-1]})")
            logger.info(f"🌐 Open in browser: file://{os.path.abspath(dashboard_file)}")
            
        except Exception as e:
            logger.error(f"Error generating dashboard for {tier}: {e}")
    
    logger.info("\n✨ All dashboards ready!")
    logger.info("\nTo view:")
    logger.info("  - Right-click on HTML file → Open with browser")
    logger.info("  - Or open browser and drag the HTML file in")


if __name__ == '__main__':
    main()

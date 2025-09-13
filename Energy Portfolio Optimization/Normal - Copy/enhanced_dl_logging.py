#!/usr/bin/env python3
"""
Enhanced DL Overlay Logging System

Adds comprehensive logging for DL hedge optimization metrics that are currently missing
from the standard CSV and diagnostic logs.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque
import logging


class DLOverlayLogger:
    """
    Comprehensive logging system for DL overlay hedge optimization metrics.
    
    Tracks:
    - Hedge performance and accuracy
    - DL model training progression  
    - Economic validation metrics
    - Feature importance and model confidence
    """
    
    def __init__(self, log_dir: str = "dl_overlay_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hedge_performance_log = os.path.join(log_dir, f"hedge_performance_{timestamp}.csv")
        self.model_training_log = os.path.join(log_dir, f"model_training_{timestamp}.csv") 
        self.economic_validation_log = os.path.join(log_dir, f"economic_validation_{timestamp}.csv")
        self.feature_importance_log = os.path.join(log_dir, f"feature_importance_{timestamp}.json")
        
        # Initialize tracking variables
        self.hedge_predictions = deque(maxlen=1000)
        self.hedge_actuals = deque(maxlen=1000)
        self.training_losses = deque(maxlen=1000)
        self.economic_metrics = deque(maxlen=1000)
        
        # Create CSV headers
        self._initialize_csv_headers()
        
        logging.info(f"DL Overlay Logger initialized: {log_dir}")
    
    def _initialize_csv_headers(self):
        """Initialize CSV file headers"""
        
        # Hedge Performance CSV
        hedge_headers = [
            'timestamp', 'step', 'timestep',
            # Predictions
            'pred_hedge_intensity', 'pred_hedge_direction', 'pred_risk_allocation_wind',
            'pred_risk_allocation_solar', 'pred_risk_allocation_hydro', 'pred_hedge_effectiveness',
            # Actuals (optimal)
            'optimal_hedge_intensity', 'optimal_hedge_direction', 'optimal_risk_allocation_wind',
            'optimal_risk_allocation_solar', 'optimal_risk_allocation_hydro', 'optimal_hedge_effectiveness',
            # Accuracy metrics
            'intensity_error', 'direction_error', 'allocation_error', 'effectiveness_error',
            'overall_accuracy', 'hedge_hit_rate',
            # Economic impact
            'dl_hedge_pnl', 'heuristic_hedge_pnl', 'dl_vs_heuristic_improvement',
            'hedge_cost', 'hedge_benefit', 'net_hedge_value'
        ]
        
        # Model Training CSV  
        training_headers = [
            'timestamp', 'step', 'epoch', 'batch',
            # Loss components
            'total_loss', 'intensity_loss', 'direction_loss', 'allocation_loss', 'effectiveness_loss',
            # Metrics
            'intensity_mae', 'direction_mae', 'allocation_accuracy', 'effectiveness_accuracy',
            # Model state
            'learning_rate', 'gradient_norm', 'model_confidence',
            # Training efficiency
            'training_time_ms', 'samples_processed', 'convergence_score'
        ]
        
        # Economic Validation CSV
        economic_headers = [
            'timestamp', 'step', 'timestep',
            # Covariance analysis
            'operational_price_covariance', 'optimal_hedge_ratio', 'correlation_strength',
            # Risk metrics
            'portfolio_volatility', 'hedge_volatility', 'hedged_portfolio_volatility',
            'risk_reduction_pct', 'sharpe_improvement',
            # Economic validation
            'hedge_efficiency', 'cost_benefit_ratio', 'economic_value_added',
            'market_regime', 'hedge_regime_appropriateness'
        ]
        
        # Write headers
        self._write_csv_header(self.hedge_performance_log, hedge_headers)
        self._write_csv_header(self.model_training_log, training_headers)
        self._write_csv_header(self.economic_validation_log, economic_headers)
    
    def _write_csv_header(self, filepath: str, headers: List[str]):
        """Write CSV header if file doesn't exist"""
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(','.join(headers) + '\n')
    
    def log_hedge_performance(self, step: int, timestep: int, predictions: Dict[str, Any], 
                            actuals: Dict[str, Any], economic_impact: Dict[str, float]):
        """Log hedge performance metrics"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract prediction values
            pred_intensity = float(predictions.get('hedge_intensity', 0.0))
            pred_direction = float(predictions.get('hedge_direction', 0.5))
            pred_allocation = predictions.get('risk_allocation', np.array([0.33, 0.33, 0.34]))
            pred_effectiveness = float(predictions.get('hedge_effectiveness', 0.8))
            
            # Extract actual/optimal values
            opt_intensity = float(actuals.get('hedge_intensity', 1.0))
            opt_direction = float(actuals.get('hedge_direction', 0.5))
            opt_allocation = actuals.get('risk_allocation', np.array([0.33, 0.33, 0.34]))
            opt_effectiveness = float(actuals.get('hedge_effectiveness', 0.8))
            
            # Calculate accuracy metrics
            intensity_error = abs(pred_intensity - opt_intensity)
            direction_error = abs(pred_direction - opt_direction)
            allocation_error = np.mean(np.abs(pred_allocation - opt_allocation))
            effectiveness_error = abs(pred_effectiveness - opt_effectiveness)
            
            overall_accuracy = 1.0 - np.mean([intensity_error, direction_error, allocation_error, effectiveness_error])
            
            # Calculate hit rate (rolling window)
            self.hedge_predictions.append(predictions)
            self.hedge_actuals.append(actuals)
            
            if len(self.hedge_predictions) >= 10:
                recent_accuracies = []
                for i in range(-10, 0):
                    pred = self.hedge_predictions[i]
                    actual = self.hedge_actuals[i]
                    acc = 1.0 - abs(pred.get('hedge_intensity', 0) - actual.get('hedge_intensity', 1))
                    recent_accuracies.append(acc > 0.8)  # Hit if accuracy > 80%
                hedge_hit_rate = np.mean(recent_accuracies)
            else:
                hedge_hit_rate = 0.8  # Default
            
            # Economic impact
            dl_pnl = economic_impact.get('dl_hedge_pnl', 0.0)
            heuristic_pnl = economic_impact.get('heuristic_hedge_pnl', 0.0)
            improvement = dl_pnl - heuristic_pnl
            hedge_cost = economic_impact.get('hedge_cost', 0.0)
            hedge_benefit = economic_impact.get('hedge_benefit', 0.0)
            net_value = hedge_benefit - hedge_cost
            
            # Write to CSV
            row = [
                timestamp, step, timestep,
                pred_intensity, pred_direction, pred_allocation[0], pred_allocation[1], pred_allocation[2], pred_effectiveness,
                opt_intensity, opt_direction, opt_allocation[0], opt_allocation[1], opt_allocation[2], opt_effectiveness,
                intensity_error, direction_error, allocation_error, effectiveness_error, overall_accuracy, hedge_hit_rate,
                dl_pnl, heuristic_pnl, improvement, hedge_cost, hedge_benefit, net_value
            ]
            
            with open(self.hedge_performance_log, 'a') as f:
                f.write(','.join(map(str, row)) + '\n')
                
        except Exception as e:
            logging.warning(f"DL hedge performance logging failed: {e}")
    
    def log_model_training(self, step: int, epoch: int, batch: int, losses: Dict[str, float],
                          metrics: Dict[str, float], training_info: Dict[str, Any]):
        """Log model training metrics"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract loss components
            total_loss = losses.get('total_loss', 0.0)
            intensity_loss = losses.get('intensity_loss', 0.0)
            direction_loss = losses.get('direction_loss', 0.0)
            allocation_loss = losses.get('allocation_loss', 0.0)
            effectiveness_loss = losses.get('effectiveness_loss', 0.0)
            
            # Extract metrics
            intensity_mae = metrics.get('intensity_mae', 0.0)
            direction_mae = metrics.get('direction_mae', 0.0)
            allocation_accuracy = metrics.get('allocation_accuracy', 0.0)
            effectiveness_accuracy = metrics.get('effectiveness_accuracy', 0.0)
            
            # Training info
            learning_rate = training_info.get('learning_rate', 0.001)
            gradient_norm = training_info.get('gradient_norm', 0.0)
            model_confidence = training_info.get('model_confidence', 0.8)
            training_time_ms = training_info.get('training_time_ms', 0.0)
            samples_processed = training_info.get('samples_processed', 0)
            
            # Calculate convergence score
            self.training_losses.append(total_loss)
            if len(self.training_losses) >= 10:
                recent_losses = list(self.training_losses)[-10:]
                convergence_score = 1.0 - (np.std(recent_losses) / (np.mean(recent_losses) + 1e-9))
            else:
                convergence_score = 0.5
            
            # Write to CSV
            row = [
                timestamp, step, epoch, batch,
                total_loss, intensity_loss, direction_loss, allocation_loss, effectiveness_loss,
                intensity_mae, direction_mae, allocation_accuracy, effectiveness_accuracy,
                learning_rate, gradient_norm, model_confidence,
                training_time_ms, samples_processed, convergence_score
            ]
            
            with open(self.model_training_log, 'a') as f:
                f.write(','.join(map(str, row)) + '\n')
                
        except Exception as e:
            logging.warning(f"DL model training logging failed: {e}")
    
    def log_economic_validation(self, step: int, timestep: int, covariance_analysis: Dict[str, float],
                              risk_metrics: Dict[str, float], validation_metrics: Dict[str, float]):
        """Log economic validation metrics"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Covariance analysis
            op_price_cov = covariance_analysis.get('operational_price_covariance', 0.0)
            optimal_ratio = covariance_analysis.get('optimal_hedge_ratio', 0.0)
            correlation = covariance_analysis.get('correlation_strength', 0.0)
            
            # Risk metrics
            portfolio_vol = risk_metrics.get('portfolio_volatility', 0.0)
            hedge_vol = risk_metrics.get('hedge_volatility', 0.0)
            hedged_vol = risk_metrics.get('hedged_portfolio_volatility', 0.0)
            risk_reduction = risk_metrics.get('risk_reduction_pct', 0.0)
            sharpe_improvement = risk_metrics.get('sharpe_improvement', 0.0)
            
            # Economic validation
            hedge_efficiency = validation_metrics.get('hedge_efficiency', 0.8)
            cost_benefit_ratio = validation_metrics.get('cost_benefit_ratio', 1.0)
            economic_value = validation_metrics.get('economic_value_added', 0.0)
            market_regime = validation_metrics.get('market_regime', 'normal')
            regime_appropriateness = validation_metrics.get('hedge_regime_appropriateness', 0.8)
            
            # Write to CSV
            row = [
                timestamp, step, timestep,
                op_price_cov, optimal_ratio, correlation,
                portfolio_vol, hedge_vol, hedged_vol, risk_reduction, sharpe_improvement,
                hedge_efficiency, cost_benefit_ratio, economic_value, market_regime, regime_appropriateness
            ]
            
            with open(self.economic_validation_log, 'a') as f:
                f.write(','.join(map(str, row)) + '\n')
                
        except Exception as e:
            logging.warning(f"DL economic validation logging failed: {e}")
    
    def save_feature_importance(self, feature_names: List[str], importance_scores: List[float],
                              model_metadata: Dict[str, Any]):
        """Save feature importance analysis"""
        try:
            feature_importance = {
                'timestamp': datetime.now().isoformat(),
                'model_metadata': model_metadata,
                'feature_importance': {
                    name: float(score) for name, score in zip(feature_names, importance_scores)
                },
                'top_features': sorted(
                    zip(feature_names, importance_scores), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }
            
            with open(self.feature_importance_log, 'w') as f:
                json.dump(feature_importance, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Feature importance logging failed: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'log_files': {
                    'hedge_performance': self.hedge_performance_log,
                    'model_training': self.model_training_log,
                    'economic_validation': self.economic_validation_log,
                    'feature_importance': self.feature_importance_log
                },
                'metrics_summary': {}
            }
            
            # Load and summarize hedge performance
            if os.path.exists(self.hedge_performance_log):
                df = pd.read_csv(self.hedge_performance_log)
                if len(df) > 0:
                    summary['metrics_summary']['hedge_performance'] = {
                        'total_predictions': len(df),
                        'average_accuracy': df['overall_accuracy'].mean(),
                        'hit_rate': df['hedge_hit_rate'].mean(),
                        'economic_improvement': df['dl_vs_heuristic_improvement'].mean(),
                        'best_accuracy': df['overall_accuracy'].max(),
                        'worst_accuracy': df['overall_accuracy'].min()
                    }
            
            # Load and summarize training metrics
            if os.path.exists(self.model_training_log):
                df = pd.read_csv(self.model_training_log)
                if len(df) > 0:
                    summary['metrics_summary']['model_training'] = {
                        'total_training_steps': len(df),
                        'final_loss': df['total_loss'].iloc[-1],
                        'loss_improvement': df['total_loss'].iloc[0] - df['total_loss'].iloc[-1],
                        'convergence_score': df['convergence_score'].iloc[-1],
                        'average_training_time': df['training_time_ms'].mean()
                    }
            
            return summary
            
        except Exception as e:
            logging.warning(f"Summary report generation failed: {e}")
            return {'error': str(e)}


# Global logger instance
_dl_logger: Optional[DLOverlayLogger] = None

def get_dl_logger(log_dir: str = "dl_overlay_logs") -> DLOverlayLogger:
    """Get or create global DL overlay logger"""
    global _dl_logger
    if _dl_logger is None:
        _dl_logger = DLOverlayLogger(log_dir)
    return _dl_logger

def log_hedge_performance(step: int, timestep: int, predictions: Dict, actuals: Dict, economic_impact: Dict):
    """Convenience function for hedge performance logging"""
    logger = get_dl_logger()
    logger.log_hedge_performance(step, timestep, predictions, actuals, economic_impact)

def log_model_training(step: int, epoch: int, batch: int, losses: Dict, metrics: Dict, training_info: Dict):
    """Convenience function for model training logging"""
    logger = get_dl_logger()
    logger.log_model_training(step, epoch, batch, losses, metrics, training_info)

def log_economic_validation(step: int, timestep: int, covariance_analysis: Dict, risk_metrics: Dict, validation_metrics: Dict):
    """Convenience function for economic validation logging"""
    logger = get_dl_logger()
    logger.log_economic_validation(step, timestep, covariance_analysis, risk_metrics, validation_metrics)

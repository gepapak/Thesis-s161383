#!/usr/bin/env python3
"""
Evaluate Trained RL Agents on New 2025 Data
Load pre-trained agents and test on unseen data
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils import load_energy_data
from RenewableMultiAgentEnv import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from multi_horizon_generator import MultiHorizonForecastGenerator
from multi_horizon_wrapper import MultiHorizonWrapperEnv

class EvaluationConfig:
    def __init__(self):
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "PPO"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "SAC"},  # meta_controller_0
        ]

def evaluate_trained_agents(data, trained_agents_dir, model_dir, scaler_dir, 
                          evaluation_steps=None, log_path=None):
    """
    Evaluate pre-trained agents on new data.
    """
    print(f"🔍 Evaluating trained agents on new data...")
    print(f"📊 Evaluation data shape: {data.shape}")
    
    # Initialize forecaster
    print(f"🔮 Loading forecaster...")
    try:
        forecaster = MultiHorizonForecastGenerator(
            model_dir=model_dir, 
            scaler_dir=scaler_dir,
            look_back=6, 
            verbose=False
        )
    except Exception as e:
        print(f"❌ Failed to load forecaster: {e}")
        return None
    
    # Setup environment
    print(f"🏗️ Setting up evaluation environment...")
    base_env = RenewableMultiAgentEnv(
        data, 
        investment_freq=144,
        forecast_generator=forecaster
    )
    
    # Create wrapper with evaluation logging
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"evaluation_logs/evaluation_metrics_{timestamp}.csv"
    
    os.makedirs("evaluation_logs", exist_ok=True)
    
    eval_env = MultiHorizonWrapperEnv(
        base_env, 
        forecaster, 
        log_path=log_path
    )
    
    # Initialize agent system for loading
    print(f"🤖 Initializing agent system...")
    config = EvaluationConfig()
    agent_system = MultiESGAgent(
        config, 
        env=eval_env, 
        device="cpu",  # Use CPU for evaluation
        training=False,  # Set to evaluation mode
        debug=False
    )
    
    # Load trained policies
    print(f"📂 Loading trained policies from {trained_agents_dir}...")
    loaded_count = agent_system.load_policies(trained_agents_dir)
    
    if loaded_count == 0:
        print(f"❌ No policies loaded! Check the path: {trained_agents_dir}")
        return None
    
    print(f"✅ Loaded {loaded_count} trained policies")
    
    # Run evaluation
    print(f"🚀 Starting evaluation...")
    
    obs, _ = eval_env.reset()
    
    # Determine evaluation length
    if evaluation_steps is None:
        evaluation_steps = min(len(data) - 1, 10000)  # Evaluate on full data or 10k steps
    
    print(f"📏 Evaluating for {evaluation_steps} steps...")
    
    # Evaluation metrics
    episode_rewards = {agent: [] for agent in eval_env.possible_agents}
    portfolio_values = []
    risk_levels = []
    actions_taken = {agent: [] for agent in eval_env.possible_agents}
    
    for step in range(evaluation_steps):
        if step % 1000 == 0:
            print(f"📊 Evaluation progress: {step}/{evaluation_steps} ({step/evaluation_steps*100:.1f}%)")
        
        # Get actions from trained policies (deterministic for evaluation)
        actions = {}
        
        for i, agent in enumerate(eval_env.possible_agents):
            if agent in obs:
                agent_obs = obs[agent].reshape(1, -1)
                
                # Get deterministic action from trained policy
                policy = agent_system.policies[i]
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(agent_obs, deterministic=True)
                    actions[agent] = action[0] if len(action.shape) > 1 else action
                else:
                    # Fallback for policies without predict method
                    action_space = eval_env.action_space(agent)
                    actions[agent] = action_space.sample()
                
                # Store action for analysis
                actions_taken[agent].append(actions[agent].copy())
        
        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)
            
            # Store metrics
            for agent, reward in rewards.items():
                episode_rewards[agent].append(reward)
            
            # Store portfolio metrics
            portfolio_value = eval_env.budget + (eval_env.wind_capacity + eval_env.solar_capacity + eval_env.hydro_capacity) * 100
            portfolio_values.append(portfolio_value)
            
            # Store risk level if available
            if hasattr(eval_env, 'market_stress'):
                risk_levels.append(eval_env.market_stress)
            
            # Check if done
            if any(dones.values()):
                print(f"💡 Episode completed at step {step}")
                break
                
        except Exception as e:
            print(f"⚠️ Error at step {step}: {e}")
            break
    
    # Calculate evaluation results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"=" * 50)
    
    # Agent performance
    for agent, rewards_list in episode_rewards.items():
        if rewards_list:
            avg_reward = np.mean(rewards_list)
            std_reward = np.std(rewards_list)
            total_reward = np.sum(rewards_list)
            
            print(f"🤖 {agent}:")
            print(f"   Average Reward: {avg_reward:.6f}")
            print(f"   Std Deviation: {std_reward:.6f}")
            print(f"   Total Reward: {total_reward:.4f}")
            print(f"   Steps: {len(rewards_list)}")
    
    # Portfolio performance
    if portfolio_values:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        max_value = max(portfolio_values)
        min_value = min(portfolio_values)
        
        roi = (final_value - initial_value) / initial_value * 100
        
        print(f"\n💰 PORTFOLIO PERFORMANCE:")
        print(f"   Initial Value: ${initial_value:,.0f}")
        print(f"   Final Value: ${final_value:,.0f}")
        print(f"   ROI: {roi:+.2f}%")
        print(f"   Max Value: ${max_value:,.0f}")
        print(f"   Min Value: ${min_value:,.0f}")
    
    # Risk management
    if risk_levels:
        avg_risk = np.mean(risk_levels)
        max_risk = max(risk_levels)
        high_risk_periods = sum(1 for r in risk_levels if r > 0.7)
        risk_pct = high_risk_periods / len(risk_levels) * 100
        
        print(f"\n⚠️ RISK MANAGEMENT:")
        print(f"   Average Risk Level: {avg_risk:.3f}")
        print(f"   Maximum Risk Level: {max_risk:.3f}")
        print(f"   High Risk Periods: {high_risk_periods} ({risk_pct:.1f}%)")
        
        if risk_pct < 10:
            print(f"   ✅ Excellent risk control")
        elif risk_pct < 25:
            print(f"   ✅ Good risk management")
        else:
            print(f"   ⚠️ Consider risk management tuning")
    
    print(f"\n📈 Detailed metrics saved to: {log_path}")
    
    # Return results for further analysis
    return {
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'risk_levels': risk_levels,
        'actions_taken': actions_taken,
        'log_path': log_path,
        'steps_evaluated': len(portfolio_values)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agents")
    parser.add_argument("--eval_data", type=str, required=True,
                       help="Path to evaluation data (new 2025 data)")
    parser.add_argument("--trained_agents", type=str, default="trained_rl_agents",
                       help="Directory with trained agent policies")
    parser.add_argument("--model_dir", type=str, default="multi_horizon_models",
                       help="Directory containing trained forecast models")
    parser.add_argument("--scaler_dir", type=str, default="multi_horizon_scalers",
                       help="Directory containing trained scalers")
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="Number of steps to evaluate (default: all data)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    print("🔍 RL Agent Evaluation on New Data")
    print("=" * 40)
    
    # Load evaluation data
    print(f"📊 Loading evaluation data: {args.eval_data}")
    try:
        eval_data = load_energy_data(args.eval_data)
        print(f"✅ Loaded evaluation data: {eval_data.shape}")
        print(f"📅 Date range: {eval_data.index[0] if hasattr(eval_data, 'index') else 'N/A'} to {eval_data.index[-1] if hasattr(eval_data, 'index') else 'N/A'}")
    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        return
    
    # Check if trained agents exist
    if not os.path.exists(args.trained_agents):
        print(f"❌ Trained agents directory not found: {args.trained_agents}")
        print(f"💡 Make sure you've trained agents first with:")
        print(f"   python main_multi_horizon.py --data_path your_2024_2025_data.csv")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup evaluation log path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"evaluation_metrics_{timestamp}.csv")
    
    # Run evaluation
    results = evaluate_trained_agents(
        data=eval_data,
        trained_agents_dir=args.trained_agents,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
        evaluation_steps=args.eval_steps,
        log_path=log_path
    )
    
    if results:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"📈 Detailed metrics: {log_path}")
        
        # Save summary results
        summary_path = os.path.join(args.output_dir, f"evaluation_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation Data: {args.eval_data}\n")
            f.write(f"Trained Agents: {args.trained_agents}\n")
            f.write(f"Steps Evaluated: {results['steps_evaluated']}\n")
            f.write(f"Portfolio Values: {len(results['portfolio_values'])}\n")
            
            if results['portfolio_values']:
                roi = (results['portfolio_values'][-1] - results['portfolio_values'][0]) / results['portfolio_values'][0] * 100
                f.write(f"Portfolio ROI: {roi:+.2f}%\n")
            
            if results['risk_levels']:
                avg_risk = np.mean(results['risk_levels'])
                f.write(f"Average Risk: {avg_risk:.3f}\n")
        
        print(f"📄 Summary saved to: {summary_path}")
        
        print(f"\n💡 Next steps:")
        print(f"   📊 Run portfolio_analyzer.py on the evaluation log")
        print(f"   📈 Compare training vs evaluation performance")
        print(f"   🔍 Analyze agent behavior on unseen data")
    
if __name__ == "__main__":
    main()
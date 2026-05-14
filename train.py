import gymnasium as gym
import torch
import numpy as np
import os
from env.reactor_env import NormanReactorEnv
from agent.sac_agent import SACAgent, ReplayBuffer
from env.demand import DemandGenerator
from scripts.dashboard import ReactorDashboard

def main():
    # Each step simulates 60 seconds of reactor physics via RK4.
    env = NormanReactorEnv(dt=60.0) 
    
    # Observation: [Flux, Iodine, Xenon, Power, Target]
    agent = SACAgent(state_dim=5, action_dim=1)
    
    memory = ReplayBuffer(capacity=100000)
    
    demand_gen = DemandGenerator()
    dashboard = ReactorDashboard()
    
    batch_size = 256
    episodes = 2000
    
    print(f"--- Norman_The_Machine ---")
    print(f"Status: Training Ground Ready")
    print(f"Device: {agent.device}")
    print(f"Hardware: Utilizing i9-14900HX High-Performance Cores")
    print(f"--------------------------")

    if not os.path.exists("data"):
        os.makedirs("data")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        hidden_state = None  # The LSTM's temporal memory
        
        step = 0
        while not done:
            target_power = demand_gen.get_target(step)
            env.target_power = target_power
            
            action, hidden_state = agent.select_action(state, hidden_state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.push(state, action, reward, next_state, done)
            
            if len(memory) > batch_size:
                agent.update(memory, batch_size)
            
            if step % 5 == 0:
                dashboard.update(step, env.engine.get_state(), target_power, action)
                
            state = next_state
            episode_reward += reward
            step += 1
            
        if ep % 10 == 0:
            print(f"Episode {ep} | Reward: {episode_reward:.2f} | Buffer: {len(memory)}")
            torch.save(agent.actor.state_dict(), "data/norman_actor_latest.pth")

if __name__ == "__main__":
    main()
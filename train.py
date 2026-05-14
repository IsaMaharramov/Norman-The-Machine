import gymnasium as gym
import torch
import numpy as np
import os
from env.reactor_env import NormanReactorEnv
from agent.sac_agent import SACAgent, ReplayBuffer
from env.demand import DemandGenerator
from scripts.dashboard import ReactorDashboard

def main():
    env = NormanReactorEnv(dt=10.0) 
    agent = SACAgent(state_dim=5, action_dim=1)
    memory = ReplayBuffer(capacity=100000)
    
    demand_gen = DemandGenerator()
    dashboard = ReactorDashboard()
    
    batch_size = 256
    episodes = 2000
    
    print(f"Status: Training Ground Ready")
    print(f"Device: {agent.device}")

    if not os.path.exists("data"):
        os.makedirs("data")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        hidden_state = None  
        
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

            if step % 100 == 0 and step > 0:
                print(f"   ... Ep {ep} | Step {step}/1440 | Current Reward: {episode_reward:.2f}")
                
            state = next_state
            episode_reward += reward
            step += 1
            

        print(f"Episode {ep} Complete | Final Reward: {episode_reward:.2f} | Memory: {len(memory)}")
        torch.save(agent.actor.state_dict(), f"data/norman_actor_latest.pth")

if __name__ == "__main__":
    main()
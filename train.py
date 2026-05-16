import gymnasium as gym
import torch
import numpy as np
import os
import sys
from env.reactor_env import NormanReactorEnv
from agent.sac_agent import SACAgent, ReplayBuffer
from scripts.dashboard import ReactorDashboard

def save_checkpoint(agent, episode, path="data/norman_checkpoint.pth"):
    torch.save({
        'episode': episode,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'critic_1_target_state_dict': agent.critic_1_target.state_dict(),
        'critic_2_target_state_dict': agent.critic_2_target.state_dict(),
        'actor_opt_state_dict': agent.actor_opt.state_dict(),
        'critic_1_opt_state_dict': agent.critic_1_opt.state_dict(),
        'critic_2_opt_state_dict': agent.critic_2_opt.state_dict(),
    }, path)

def main():
    env = NormanReactorEnv(dt=10.0) 
    agent = SACAgent(state_dim=5, action_dim=1)
    memory = ReplayBuffer(capacity=100000)
    dashboard = ReactorDashboard()
    
    batch_size = 1024 
    episodes = 2000
    start_episode = 0
    checkpoint_path = "data/norman_checkpoint.pth"

    print(f"Status: Training Ground Ready")
    print(f"Device: {agent.device} [GPU Activated]")
    
    if not os.path.exists("data"):
        os.makedirs("data")

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint. Loading...")
        cp = torch.load(checkpoint_path, map_location=agent.device)
        
        if 'actor_state_dict' in cp:
            agent.actor.load_state_dict(cp['actor_state_dict'])
            agent.critic_1.load_state_dict(cp['critic_1_state_dict'])
            agent.critic_2.load_state_dict(cp['critic_2_state_dict'])
            if 'actor_opt_state_dict' in cp:
                agent.critic_1_target.load_state_dict(cp['critic_1_target_state_dict'])
                agent.critic_2_target.load_state_dict(cp['critic_2_target_state_dict'])
                agent.actor_opt.load_state_dict(cp['actor_opt_state_dict'])
                agent.critic_1_opt.load_state_dict(cp['critic_1_opt_state_dict'])
                agent.critic_2_opt.load_state_dict(cp['critic_2_opt_state_dict'])
        else:
            agent.actor.load_state_dict(cp['actor_state'])
            agent.critic_1.load_state_dict(cp['critic1_state'])
            agent.critic_2.load_state_dict(cp['critic2_state'])
            
        start_episode = cp.get('episode', 0) + 1
        print(f"Successfully loaded! Resuming from Episode {start_episode}")

    try:
        for ep in range(start_episode, episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            hidden_state = None  
            
            step = 0
            while not done:
                target_power = env.target_power 
                
                action, hidden_state = agent.select_action(state, hidden_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                memory.push(state, action, reward, next_state, done)
                
                if len(memory) > batch_size:
                    agent.update(memory, batch_size)
         
                if ep % 5 == 0 and step % 5 == 0:
                    try:
                        dashboard.update(step, env.engine.get_state(), target_power, action)
                    except Exception:
                        pass

                if step % 100 == 0 and step > 0:
                    print(f"   ... Ep {ep} | Step {step}/1440 | Current Reward: {episode_reward:.2f}")
                    
                state = next_state
                episode_reward += reward
                step += 1
                
            print(f"Episode {ep} Complete | Final Reward: {episode_reward:.2f} | Memory: {len(memory)}")
            
            if ep % 2 == 0:
                save_checkpoint(agent, ep, checkpoint_path)

    except KeyboardInterrupt:
        print("\nManual Interruption Detected.")
        save_checkpoint(agent, ep, checkpoint_path)
        print("Emergency Checkpoint Saved. Safe to close.")
        sys.exit(0)

if __name__ == "__main__":
    main()
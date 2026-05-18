import torch
import time
from envs.reactor_env import NormanReactorEnv
from agent.sac_agent import SACAgent
from scripts.dashboard import ReactorDashboard

def main():
    env = NormanReactorEnv(dt=10.0) 
    agent = SACAgent(state_dim=5, action_dim=1)
    dashboard = ReactorDashboard()
    
    checkpoint_path = "data/norman_checkpoint.pth"
    
    try:
        cp = torch.load(checkpoint_path, map_location=agent.device)
        agent.actor.load_state_dict(cp['actor_state_dict'])
        print("Starting reactor...")
    except FileNotFoundError:
        print("Error: Could not find norman_checkpoint.pth. Did you save it in the data folder?")
        return

    state, _ = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    while not done:
        target_power = env.target_power 
        
        action = agent.select_action(state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if step % 5 == 0:
            try:
                dashboard.update(step, env.engine.get_state(), target_power, action)
            except Exception:
                pass
        
        state = next_state
        total_reward += reward
        step += 1
        
        time.sleep(0.01)
        
    print(f"Simulation Complete! Final Evaluation Score: {total_reward:.2f}")
    
    input("Press Enter to close the dashboard...")

if __name__ == "__main__":
    main()
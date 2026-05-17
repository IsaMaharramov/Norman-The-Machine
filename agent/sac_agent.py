import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from agent.networks import Actor, Critic

class ReplayBuffer:
    def __init__(self, capacity, state_dim=5, action_dim=1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
    
    
    def push(self, state, action, reward, next_state, done):
        try:
            self.state[self.ptr] = np.array(state, dtype=np.float32).reshape(5)
            self.action[self.ptr] = np.array(action, dtype=np.float32).reshape(1)
            self.reward[self.ptr] = reward
            self.next_state[self.ptr] = np.array(next_state, dtype=np.float32).reshape(5)
            self.done[self.ptr] = done
            
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        except Exception:
            pass
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

    def __len__(self):
        return self.size

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        if np.any(np.isnan(state)):
            return np.array([0.0])

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mu, log_std = self.actor(state_t)
            
            if torch.isnan(mu).any():
                return np.array([0.0])

            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = torch.tanh(dist.rsample())
            
        return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size=1024):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_mu, next_log_std = self.actor(ns)
            next_dist = torch.distributions.Normal(next_mu, next_log_std.exp())
            next_actions_raw = next_dist.rsample()
            next_actions_3d = torch.tanh(next_actions_raw)
            
            q1_t = self.critic_1_target(ns, next_actions_3d)
            q2_t = self.critic_2_target(ns, next_actions_3d)
            
            log_prob_next = next_dist.log_prob(next_actions_raw).sum(-1, keepdim=True)
            target_v = torch.min(q1_t, q2_t) - self.alpha * log_prob_next
            target_q = r + (1 - d) * self.gamma * target_v

        curr_q1 = self.critic_1(s, a)
        curr_q2 = self.critic_2(s, a)
        
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.critic_1_opt.zero_grad()
        q1_loss.backward()
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        q2_loss.backward()
        self.critic_2_opt.step()

        mu, log_std = self.actor(s)
        dist = torch.distributions.Normal(mu, log_std.exp())
        curr_a_raw = dist.rsample()
        curr_a_3d = torch.tanh(curr_a_raw)
        
        q1 = self.critic_1(s, curr_a_3d)
        q2 = self.critic_2(s, curr_a_3d)
        min_q = torch.min(q1, q2)
        
        log_prob = dist.log_prob(curr_a_raw).sum(-1, keepdim=True)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


'''
        s = torch.FloatTensor(states).unsqueeze(1).to(self.device)        # (B, 1, 5)
        a = torch.FloatTensor(actions).view(batch_size, 1, 1).to(self.device) # (B, 1, 1)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)       # (B, 1)
        ns = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)  # (B, 1, 5)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)         # (B, 1)


'''
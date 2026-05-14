import torch
import torch.nn.functional as F
import torch.optim as optim
from agent.networks import Actor, Critic
import numpy as np

class SACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # temperature for entropy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor and Twin Critics
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        
        # Target Networks -> stability
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state, hidden_state=None):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        mu, log_std, next_hidden = self.actor(state, hidden_state)
        std = log_std.exp()
        
        # Reparameterization trick
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()[0], next_hidden

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # to -> tensors
        s = torch.FloatTensor(states).unsqueeze(1).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_mu, next_log_std, _ = self.actor(ns)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mu, next_std)
            next_actions = torch.tanh(next_dist.rsample())
            
            # Twin-Q targets to prevent overestimation
            q1_t, _ = self.critic_1_target(ns, next_actions)
            q2_t, _ = self.critic_2_target(ns, next_actions)
            target_v = torch.min(q1_t, q2_t) - self.alpha * next_dist.log_prob(next_actions).sum(-1, keepdim=True)
            target_q = r + (1 - d) * self.gamma * target_v

        # MSE Loss for Critics
        curr_q1, _ = self.critic_1(s, a)
        curr_q2, _ = self.critic_2(s, a)
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        self.critic_1_opt.zero_grad()
        q1_loss.backward()
        self.critic_1_opt.step()

        self.critic_2_opt.zero_grad()
        q2_loss.backward()
        self.critic_2_opt.step()


        mu, log_std, _ = self.actor(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        curr_a = torch.tanh(dist.rsample())
        
        q1, _ = self.critic_1(s, curr_a)
        q2, _ = self.critic_2(s, curr_a)
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * dist.log_prob(curr_a).sum(-1, keepdim=True) - min_q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from replay_buffer import ReplayBuffer


class Net(nn.Module):
    def __init__(self,
                 action_dim,
                 state_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,action_dim)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self,
                 action_dim,
                 state_dim,
                 learning_rate,
                 batch_size,
                 epsilon,
                 epsilon_decay,
                 discount_rate,
                 buffer_capacity,
                 target_network_update_freq,
                 device):
        self.net = Net(action_dim, state_dim).to(device)
        self.target_net = Net(action_dim, state_dim).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.action_dim = action_dim
        self.observation_dim = state_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.device = device
        self.target_network_update_freq = target_network_update_freq
        self.learn_step_counter = 0
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        """选择动作"""
        state = torch.Tensor(state).to(self.device)
        if np.random.rand() >= self.epsilon:
            with torch.no_grad():
                action_value = self.net.forward(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.choice(self.action_dim)

        if self.epsilon >= 0.05:
            self.epsilon -= self.epsilon_decay
        return action

    def store_transition(self,
                         state,
                         action,
                         reward,
                         next_state,
                         done):
        """添加训练数据"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """训练模型"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions).reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(self.device)

        # 计算TD error
        q_eval = self.net(states).gather(1, actions)
        q_next  = self.target_net(next_states).max(1)[0].detach().reshape(-1, 1)
        q_target = rewards + (1-dones) * self.discount_rate * q_next
        loss = self.loss_func(q_eval, q_target)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_network_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return loss


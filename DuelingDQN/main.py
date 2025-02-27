import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from dueling_dqn import DuelingDQN

EPISODES = 600
ACTION_DIM = 2
STATE_DIM = 4
LEARNING_RATE = 9e-5
BATCH_SIZE = 128
DISCOUNT_RATE = 0.99
EPSILON = 0.9
EPSILON_DECAY = 0.001
BUFFER_CAPACITY = 1000
TARGET_UPDATE_FREQ = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建环境&智能体
env = gym.make('CartPole-v1',render_mode=None)
agent = DuelingDQN(action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            learning_rate=LEARNING_RATE,
            discount_rate=DISCOUNT_RATE,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            epsilon_decay=EPSILON_DECAY,
            target_network_update_freq=TARGET_UPDATE_FREQ,
            device=DEVICE)

reward_list = []

for i in range(EPISODES):
    state,_ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward


        if agent.replay_buffer.is_full():
            loss = agent.learn()

    reward_list.append(episode_reward)

    print(f"episode: {i}, reward: {episode_reward}")


plt.plot(reward_list)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.title('Dueling DQN CartPole-v1')
plt.show()

np.save('arr1', reward_list)

env.close()




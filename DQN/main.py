import gymnasium as gym
import torch
from numpy.lib.format import BUFFER_SIZE
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from  replay_buffer import  ReplayBuffer
from dqn import DQN
from ac import ActorCritic

EPISODES = 1000
ACTION_DIM = 4
STATE_DIM = 8
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
DISCOUNT_RATE = 0.99
EPSILON = 0.9
EPSILON_DECAY = 0.001
BUFFER_CAPACITY = 1000
TARGET_UPDATE_FREQ = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建环境&智能体
env = gym.make('LunarLander-v3',render_mode=None)
agent = ActorCritic(action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            learning_rate=LEARNING_RATE,
            discount_rate=DISCOUNT_RATE,
            epsilon=EPSILON,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            )

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
plt.title('DQN-LunarLander-v3')
plt.show()

env.close()




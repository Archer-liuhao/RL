from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states,actions, rewards, next_states, dones = [], [], [], [], []
        for experience in experiences:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones


    def is_full(self):
        return len(self.memory) == self.capacity
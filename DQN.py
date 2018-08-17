# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from RLModel import Inventory

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # temp = []
        # for x in range(action_size):
        #     if x not in inventory.actions_chosen:
        #         temp.append(x)
        #
        # i = random.randrange(len(temp))
        # return temp[i]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    import Data

    nItems = Data.n
    budget = Data.budget
    req = Data.req
    costPerItem = Data.costPerItem
    importance = Data.importance

    inventory = Inventory(nItems)
    inventory.reset(budget, req, costPerItem, importance)

    state_size = np.shape(np.concatenate((inventory.req, inventory.costPerItem, inventory.importance)))[0]
    action_size = inventory.nItems

    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    t = 1000

    rewards = np.zeros((EPISODES, t))

    for e in tqdm(range(EPISODES)):
        state = inventory.reset(10000, np.array([3, 1, 8, 9]), np.array([300, 100, 400, 700]), np.array([2, 7, 1, 2]))
        state = np.reshape(state, [1, state_size])
        print(state)
        print("------------------------\n")
        for time in range(t):
            action = agent.act(state)
            inventory.actions_chosen.append(action)
            next_state, reward, done = inventory.step(action)
            if done:
                break
            rewards[e, time] = reward
            reward = reward if not done else -10
            print(action, reward, next_state, inventory.budget)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # if done:
            # print("episode: {}/{}, score: {}, e: {:.2}"
            #       .format(e, EPISODES, time, agent.epsilon))
            # break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

plt.figure(figsize=(10, 10))
plt.xlabel('episode')
plt.ylabel('average reward')
rewards = rewards.mean(axis=1)
plt.plot(rewards)
plt.savefig("./fig1.jpg")

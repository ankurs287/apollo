import matplotlib

matplotlib.use('Agg')
import numpy as np


class Inventory:

    def __init__(self, nItems):
        self.nItems = nItems
        self.req = np.zeros(self.nItems)
        self.costPerItem = np.zeros(self.nItems)
        self.importance = np.zeros(self.nItems)
        self.state = np.concatenate((self.req, self.costPerItem, self.importance))
        self.actions = np.arange(self.nItems)

    def reset(self, budget, req, costPerItem, importance):
        self.req = req
        self.costPerItem = costPerItem
        self.importance = importance
        self.budget = budget
        self.state = np.concatenate((self.req, self.costPerItem, self.importance))
        return self.state

    # def act(self):
    #     return 1

    # Given an action, return the reward
    def step(self, action):
        self.budget -= self.req[action] * self.costPerItem[action]
        self.req[action] = 0
        self.importance[action] = 0
        self.reward = -(np.sum(self.importance)) + self.budget
        self.nextState = np.concatenate((self.req, self.costPerItem, self.importance))
        return self.nextState, self.reward, self.done()

    def done(self):
        if self.budget <= 0 or np.sum(self.req) == 0 or self.budget < np.min(self.costPerItem):
            return True
        return False

    # def simulate(self):
    #     return True

# if __name__ == "__main__":

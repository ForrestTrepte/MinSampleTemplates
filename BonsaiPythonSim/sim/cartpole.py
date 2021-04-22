import math
import random
from collections import namedtuple

class CartPole:
    def __init__(self):
        self.reset({ 'config_val': 0})

    def reset(self, config):
        self.config = config;
        self.action = { 'action_val': 0 }
        self.total = 0

    def step(self, action):
        self.action = action

    @property
    def state(self):
        self.total += self.action['action_val']

        state = { 'total': self.total}
        for config_key, config_value in self.config.items():
            state['state_' + config_key] = config_value
        for action_key, action_value in self.action.items():
            state['state_' + action_key] = action_value
        return state
"""
File to complete. Contains the agents
VEvalTemporalDifferencing and VEvalMonteCarlo (TD0)
"""
import numpy as np
import math


class Policy(object):
    """ Base class for policies. Do not modify
    """

    def __init__(self):
        super(Policy, self).__init__()

    def action(self, mdp, state, values):
        pass


class RandomPolicy(Policy):
    def __init__(self):
        super(RandomPolicy, self).__init__()

    def action(self, *args, **kwargs):
        actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return actions[np.random.choice(range(len(actions)))]


class VEvalTemporalDifferencing(object):
    def __init__(self, mdp, policy, *args, **kwargs):
        super(VEvalTemporalDifferencing, self).__init__()
        self.mdp = mdp
        self.policy = policy()
        self.values = np.zeros(mdp.size)  # Store state values in this variable
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.discount = kwargs.get('discount', 0.6)
        

    def update(self):
        
        # TO IMPLEMENT
        # Ingredients : discount, values, learning_rate, old position, new position, reward,...
        #
        self.values[self.last_position]+=self.learning_rate*(self.discount*self.values[self.mdp.position]-self.values[self.last_position]+self.mdp.grid[self.last_position])
    

    def action(self):
        self.last_position = self.mdp.position
        self.last_action = self.policy.action(
            self.mdp, self.last_position, self.values)
        return self.last_action


class VEvalMonteCarlo(object):
    def __init__(self, mdp, policy, *args, **kwargs):
        super(VEvalMonteCarlo, self).__init__()
        self.mdp = mdp
        self.policy = policy
        self.values = np.zeros(mdp.size)  # Store state values in this variable
        self.sum_values = np.zeros(mdp.size)
        self.n_transitions = np.zeros(mdp.size)
        self.discount = kwargs.get('discount', 0.6)

    def update(self):
        # TO IMPLEMENT
        # Ingredients: Reward, history of positions, values, discount,...
        pass
    

    def action(self):
        self.last_position = self.mdp.position
        self.last_action = self.policy.action(
            self.mdp, self.last_position, self.values)
        return self.last_action

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
        #Direct application of the formula
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
        #Test if we are just about to finish an episode, otherwise we don't update anything
        if(self.mdp.reward[-1][-1]!=-1):
            #Generate the list of all possible states
            states= [(x,y) for x in range(self.mdp.grid.shape[0]) for y in range(self.mdp.grid.shape[1])]
            for state in states:
                value=0 #Initialization of the value function of the state
                nb_episodes=0 #Initialization of the number of episodes where the state figues
                for episode in self.mdp.history:#Iterate through all episodes
                    if state in episode:
                        nb_episodes+=1
                        idx=episode.index(state)#Find the first utilization of the state in the episode
                        while idx<len(episode):#Sum of rewards starting from the current state
                            value+=pow(self.discount,idx)*self.mdp.grid[episode[idx]]
                            idx+=1
                if nb_episodes==0:
                    self.values[state]=0
                else:
                    self.values[state]=value/nb_episodes

    def action(self):
        self.last_position = self.mdp.position
        self.last_action = self.policy.action(
            self.mdp, self.last_position, self.values)
        return self.last_action

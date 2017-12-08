"""
File to complete. Contains the agents
VEvalTemporalDifferencing and VEvalMonteCarlo (TD0)
"""
import numpy as np
import math


class Agent(object):
    # DO NOT MODIFY
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(Agent, self).__init__()
        if initial_policy is not None:
            self.policy = initial_policy
        else:
            self.policy = np.zeros((4, mdp.size[0], mdp.size[1])) + 0.25
        # Init the random policy
        # dim[0] is the actions, in the order (up,down,left,right)
        self.mdp = mdp
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.discount = kwargs.get("discount", 0.95)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        # For some agents : V Values
        self.V = np.zeros(mdp.size)

        # For others: Q values
        self.Q = np.zeros((4, mdp.size[0], mdp.size[1]))

    def update(self):
        # DO NOT MODIFY
        raise NotImplementedError

    def action(self):#vérifier que c'est ce choix est aléatoire
        self.last_position = self.mdp.position
        return self.actions[np.random.choice(range(len(self.actions)),
                                             p=self.policy[:, self.last_position[0],
                                                           self.last_position[1]])]


class ValueIteration(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(ValueIteration, self).__init__(
            mdp, initial_policy, *args, **kwargs)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(ValueIteration, self).action()


class PolicyIteration(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(PolicyIteration, self).__init__(
            mdp, initial_policy, *args, **kwargs)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(PolicyIteration, self).action()


class QLearning(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(QLearning, self).__init__(mdp, initial_policy, *args, **kwargs)


    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.Q[a,state2[0],state2[1]] for a in range(4)])
        self.learnQ(state1, action1, reward, reward + self.discount*maxqnew)

    def learnQ(self, state, action, reward, value):
        oldv = self.Q[action,state[0],state[1]]
        self.Q[action,state[0],state[1]] = oldv + self.learning_rate * (value - oldv)

    def update(self):
        # TO IMPLEMENT
        self.learn(self.last_position, self.last_action, self.mdp.reward[-1][-1], self.mdp.position)

    def action(self):
        # YOU CAN MODIFY
        state=self.mdp.position
        if(np.random.uniform(0,1)<0.1):
            action=np.random.choice(range(4))
        else:         
            q = [self.Q[a, state[0],state[1]] for a in range(4)]
            action=np.argmax(q)
        self.last_position=state
        self.last_action=action
        action = self.actions[action]
        return action

class SARSA(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(SARSA, self).__init__(mdp, initial_policy, *args, **kwargs)
	
    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.Q[action2,state2[0],state2[1]]
        self.learnQ(state1, action1, reward, reward + self.discount * qnext)
        
    def learnQ(self, state, action, reward, value):
        oldv = self.Q[action,state[0],state[1]]
        self.Q[action,state[0],state[1]] = oldv + self.learning_rate * (value - oldv)

    def update(self):
        # TO IMPLEMENT
        self.learn(self.last_last_position, self.last_last_action, self.mdp.reward[-1][-1], self.last_position,self.last_action)

    def action(self):
        # YOU CAN MODIFY

        state=self.mdp.position
        if(np.random.uniform(0,1)<0.1):
            action=np.random.choice(range(4))
        else:         
            q = [self.Q[a, state[0],state[1]] for a in range(4)]
            action=np.argmax(q)
        if(self.mdp.reward[-1]==[0]):
            self.last_position=self.mdp.position
            self.last_action=action
        self.last_last_position=self.last_position
        self.last_last_action=self.last_action
        self.last_position=state
        self.last_action=action
        action = self.actions[action]
        return action

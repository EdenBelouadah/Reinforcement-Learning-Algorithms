"""
File to complete. Contains the agents
VEvalTemporalDifferencing and VEvalMonteCarlo (TD0)
"""
import numpy as np
import math
import copy

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
        #for Epsilon greedy step:
        self.epsilon=0.1
        self.beta=0.9
        self.states= [(x,y) for x in range(self.mdp.grid.shape[0]) for y in range(self.mdp.grid.shape[1])]
        
    def normalize(self,Q):
        P=copy.deepcopy(Q)
        for a in range(P.shape[0]):
            mini=np.min(P[a])
            maxi=np.max(P[a])
            for i in range(P.shape[1]):
                for j in range(P.shape[2]):
                    P[a,i,j]=(P[a,i,j]-mini)/(maxi-mini)
        return P  
        
        

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
        
        self.stop=False

    def state_neighbors(self,state):
        condidates=[(state[0]+1,state[1]),(state[0]-1,state[1]),(state[0],state[1]+1),(state[0],state[1]-1)]
        neighbors=[condidate for condidate in condidates if ((condidate not in self.mdp.walls)and(condidate[0]>=0 and condidate[1]>=0  and condidate[0]<=9 and condidate[1]<=9))]
        return neighbors


    def update(self):
        if(self.stop==False):
            prev_v = copy.deepcopy(self.V)
            for state in self.states:
                neighbors=self.state_neighbors(state)
                self.V[state]=self.mdp.reward[-1][-1]+self.discount*(np.max([sum([(1-self.mdp.stochasticity)*prev_v[s_] for s_ in neighbors]) for a in range(4)])) 
            if (np.sum(np.fabs(prev_v - self.V)) == 0):
                print ('Value-iteration converged')  
                self.stop=True
            
    def action(self):
        state=self.mdp.position
        if(np.random.uniform(0,1)<self.epsilon):
            action=np.random.choice(range(4))
        else:         
           # action=action_to_best_state
           action=np.random.choice(range(4))
        self.last_position=state
        #self.last_action=action
        action = self.actions[action]
        return action


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
        self.V=None
       

    def update(self):
        state1=self.last_position #récupérer l'état précédent
        action1=self.last_action #récupérer l'action qui nous a mené à cet état
        reward=self.mdp.reward[-1][-1]#récupérer la reward instantanée
        state2=self.mdp.position #récupérer l'état courant
        maxqnew = max([self.Q[a,state2[0],state2[1]] for a in range(4)])#maximum des Q valeurs ) partir de l'état courant
        oldv = self.Q[action1,state1[0],state1[1]]#récupérer la Q valeur de l'action qui nous a mené à cet état
        self.Q[action1,state1[0],state1[1]] = oldv + self.learning_rate * (reward + self.discount*maxqnew - oldv)#mettre à jour la Q valeur de l'état précédent
        self.policy=self.normalize(self.Q)#normalisation de la Q valeurs et afectation à politique

    def action(self):
        state=self.mdp.position
        if(np.random.uniform(0,1)<self.epsilon):#epsilon greedy
            action=np.random.choice(range(4))#choisir une action aléatoire
            self.epsilon*=self.beta#mettre à jour epsilon
        else:         
            q = [self.Q[a, state[0],state[1]] for a in range(4)]
            action=np.argmax(q)#choisir la meilleure action
        self.last_position=state #sauvegarder l'état courant
        self.last_action=action #sauvegarder l'action prise
        action = self.actions[action]        
        return action#retourner l'action

class SARSA(Agent):
    def __init__(self, mdp, initial_policy=None, *args, **kwargs):
        super(SARSA, self).__init__(mdp, initial_policy, *args, **kwargs)
        self.V=None
        
    def update(self):
        state1=self.last_last_position
        action1=self.last_last_action
        reward=self.mdp.reward[-1][-1]
        state2=self.last_position
        action2=self.last_action
        qnext = self.Q[action2,state2[0],state2[1]]
        oldv = self.Q[action1,state1[0],state1[1]]
        self.Q[action1,state1[0],state1[1]] = oldv + self.learning_rate * (reward + self.discount * qnext - oldv)
        self.policy=self.normalize(self.Q)

    def action(self):
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
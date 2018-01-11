import sys

import pylab as plb
import numpy as np
import mountain_car


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass


# implement your own agent here

class MyAgent():
    def __init__(self, p=10, k=10, discount=0.6, learning_rate=0.1, epsilon=1):
        """
        Initialize your internal state
        """
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.p = p
        self.k = k
        self.s = np.empty((p + 1, k + 1), dtype=tuple)
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                self.s[i, j
                ] = (-150 + i * 150 / p, -20 + j * 40 / k)
        self.state = (np.random.choice(1,[p+1, k+1]), 0)
        self.W = np.zeros((3, p+1, k+1))
        self.Q = np.zeros((3, p + 1, k + 1))


    def calculate_phi(self, x, vx, s):
        phi = np.empty(self.p + 1, self.k + 1)
        for i in range(self.phi.shape[0]):
            for j in range(self.phi.shape[1]):
                self.phi[i, j] = np.exp(-(x - s[i, j][0]) ** 2) * np.exp(-(vx - s[i, j][1]) ** 2)
        return phi

    def act(self):
        """
        Choose action depending on your internal state
        """
        rand = np.random.rand()
        if rand < self.epsilon:
            action = np.random.randint(-1, 2)
        else:
            action = np.argmax([self.Q[a, self.state[0], self.state[1]] for a in range(3)])
        self.last_action = action
        return action

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        x = next_state[0]
        vx = next_state[1]
        oldv = self.Q[self.last_action, self.state[0], self.state[1]]
        maxqnew = max([self.Q[a,next_state[0], next_state[1]] for a in range(3)])#maximum des Q valeurs ) partir de l'état courant

        difference = reward + self.discount * maxqnew - oldv
        self.Q[self.last_action, self.state[0], self.state[1]] += self.learning_rate * difference
        self.w += self.learning_rate * difference * self.phi
        self.state = next_state

# test class, you do not need to modify this class
class Tester:
    def __init__(self, agent):
        self.mountain_car = mountain_car.MountainCar()
        self.agent = agent

    def visualize_trial(self, n_steps=100):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountain_car.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            print("Enter to continue...")
            input()

            sys.stdout.flush()

            reward = self.mountain_car.act(self.agent.act())
            self.agent.state = [self.mountain_car.x, self.mountain_car.vx]

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """
        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            self.mountain_car.reset()
            step = 1
            while step <= max_episode or max_episode <= 0:
                reward = self.mountain_car.act(self.agent.act())
                self.agent.update([self.mountain_car.x, self.mountain_car.vx],
                                  reward)
                rewards[c_episodes] += reward
                if reward > 0.:
                    break
                step += 1
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step - 1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    agent = MyAgent()
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(10, 20)

    print("End of learning, press Enter to visualize...")
    input()
    test.visualize_trial()
    plb.show()

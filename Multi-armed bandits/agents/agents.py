import numpy as np
import math
import random

class epsGreedyAgent:
    def __init__(self, A, epsilon):
        self.epsilon = epsilon #Probabilité d'exploration
        self.A = A #Nombre de bras
        self.Q = np.zeros(A) #Initialisation de la Q_valeur de chaque bras
        self.numpick = np.zeros(A) #Initialisation du nombre de choix de chaque bras

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon: #Exploitation
            a = np.random.randint(0, self.A) #Choisir bras aléatoire
        else:
            a = np.argmax(self.Q) #Choisir le bras ayant la meilleure Q_valeur
        return a

    def update(self, a, r):
        self.numpick[a] += 1 #Incrémenter le nombre de choix du bras choisis
        self.Q[a] += (r - self.Q[a]) / self.numpick[a] #Mise à jour de la Q_valeur de chaque bras


class optimistEpsGreedyAgent(epsGreedyAgent):
    def __init__(self, A, epsilon, optimism):
        self.epsilon = epsilon  # Probabilité d'exploration
        self.A = A  # Nombre de bras
        self.Q = np.ones(A)*optimism  # Initialisation de la Q_valeur de chaque bras
        self.numpick = np.zeros(A)  # Initialisation du nombre de choix de chaque bras

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon: #Exploitation
            a = np.random.randint(0, self.A) #Choisir bras aléatoire
        else:
            a = np.argmax(self.Q) #Choisir le bras ayant la meilleure Q_valeur
        return a

    def update(self, a, r):
        self.numpick[a] += 1 #Incrémenter le nombre de choix du bras choisis
        self.Q[a] += (r - self.Q[a]) / self.numpick[a] #Mise à jour de la Q_valeur de chaque bras

class softmaxAgent:
    def __init__(self, A, temperature):
        self.A = A #Nombre de bras
        self.Q = np.zeros(A) #Initialisation des Q_valeurs de chaque bras
        self.numpick = np.zeros(A) #Initialisation du nombre de choix de chaque bras
        self.temperature = temperature #Température

    def interact(self):
        exps = np.exp(self.Q/self.temperature - np.max(self.Q/self.temperature)) #Calculer la version stable de softmax
        somme_exps = np.sum(exps) #Calculer la somme
        exps /= somme_exps #Normalisation
        a = random.choices(range(0, self.A), exps) #Choisir le bras avec la probabilité correspondante
        return a

    def update(self, a, r):
        self.numpick[a] += 1 #Incrémenter le nombre de choix du bras choisis
        self.Q[a] += (r - self.Q[a]) / self.numpick[a] #Mise à jour de la Q_valeur de chaque bras


class UCBAgent:
    def __init__(self, A):
        self.A = A #Nombre de bras
        self.Q = np.zeros(A) #Initialisation des Q_valeurs de chaque bras
        self.numpick = np.ones(A) #Initialisation à 1 du nombre de choix de chaque bras pour éviter un dénominateur nul

    def interact(self):
        numpick = np.copy(self.numpick) #Copier le nombre de fois où chaque bras a été choisi
        numpick = np.sqrt(2 * np.log(np.sum(numpick)) / numpick) #Calculer l'intervalle de confiance + mean reward pour chaque bras
        a = np.argmax(self.Q + numpick) #Choisir le bras maximisante la somme des deux
        return a

    def update(self, a, r):
        self.numpick[a] += 1 #Incrémenter le nombre de choix du bras choisis
        self.Q[a] += (r - self.Q[a]) / self.numpick[a] #Mise à jour de la Q_valeur pour chaque bras


class ThompsonAgent:
    def __init__(self, A, mu_0, var_0):
        self.A = A #Nombre de bras
        self.Q = np.zeros(A) #Initialisation des Q_valeurs de chaque bras
        self.numpick = np.zeros(A) #Initialisation du nombre de choix de chaque bras
        self.mu = np.ones(A) * mu_0 #Moyenne de la gaussienne
        self.var = np.ones(A) * var_0 #Variance de la gaussienne
        self.samples = self.var * np.random.randn(A) + self.mu #échantillonnage initial

    def interact(self):
        self.mu = self.Q
        self.var = 1/(self.numpick+1)
        self.samples = self.var *(np.random.randn(self.A)) + self.mu #Echantillonnage selon la gaussienne
        a = np.argmax(self.samples) #Choisir le meilleur bras
        return a

    def update(self, a, r):
        self.numpick[a] += 1 #Incrémenter le nombre de choix du bras choisis
        self.Q[a] += (r - self.Q[a]) / self.numpick[a] #Mise à jour de la Q_valeur de chaque bras
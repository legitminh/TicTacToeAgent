import numpy as np
from Agent import Agent
import json
from const import *
from Environment import Environment
import math
import random
"""
Goal
    Learn from playing only, think ahead 
Agent purpose
    Win Tic Tac Toe
    Earn maximum reward
Each action
    Train a game action upon final reward is realized

Each layer train the next
"""

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid_x(x, providing_s=False):
    if providing_s:
        return x * (1-x)
    s = sigmoid(x)
    return s * (1-s)

def softmax(z):
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

def d_softmax(y):
    return y
    # return np.diag(y) - np.outer(y, y)

func_to_d_func = {
    sigmoid: d_sigmoid_x,
    # softmax: d_softmax
}
# def softmax(x):
#     """Compute softmax values for each set of scores in x."""
#     # Subtract the maximum value for numerical stability
#     e_x = np.exp(x - np.max(x)) 
#     return e_x / e_x.sum(axis=0)

chains = []
np.random.seed(None)

networkSizes = (BOARD_WIDTH * BOARD_HEIGHT, # input(environmentRelativeArray)
                16,
                32,
                BOARD_WIDTH * BOARD_HEIGHT)

neuralLayerCount = len(networkSizes) - 1 #not counting the input layer

class AgentNN_simplified(Agent):
    def __init__(self, index: int, environment : Environment):
        self.processing_functions = (
            sigmoid,
            sigmoid,
            sigmoid,
        )
        self.index = index
        self.environment = environment
        self.learningRate = 0.01 #scales the amount to move in each input dimension
        self.depreciation = 0.5
        self.explorationRate = 0.2 #chaoticity
        self.exploration_decay = 0.999
        super().__init__()
        self.weights = [
            np.random.randn(networkSizes[i+1], networkSizes[i]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.biases = [
            np.random.randn(networkSizes[i+1]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.history_forwards = [] #history of forward values for back-prop
        # self.episodeActions = []

    def loss(self, y, better_y):
        return np.sum((y - better_y)**2)

    def d_loss(self, y, better_y):
        # return 2 * (x - better_y)
        return 2 * (y - better_y) #cross entropy
    
    def forward(self, x):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description

        :return: all_x, all_z, y
        
        run through the neural network 

        x = self.weights @ x + b
        is the same with rewriting an array x where each element is equal to w_x * x + b

        return all_x which is array of all input vectors of all layers and y (output) 
        return all_z which is all output after processing of all layers
        """
        ...

    def backpropagate(self, all_x, all_z, better_y):
        """
        return Gradient to shape the network towards goal
        d_l is change in loss, d_l_[?] represent change in loss with respect to variable ?
        """
        
        d_l_f = self.d_loss(all_x[-1], better_y) # this is dl/df
        d_biases = []
        d_weights = []
        for layerIndex in range(neuralLayerCount -1, -1, -1):
            """
            f(wx+b)=f(z)=x1
            dl_z = dl_f * df_z : z input
            dl_x = dl_z * dz_x
            """
            z = all_z[layerIndex] # z of this layer
            f = all_x[layerIndex+1] # f of this layer
            d_l_z = d_l_f * func_to_d_func[self.processing_functions[layerIndex]](f, providing_s=True)

            d_l_b = d_l_z
            d_biases.insert(0, d_l_b)

            prev_x = all_x[layerIndex] # x of prev layer
            d_l_w = np.outer(d_l_z, prev_x)
            d_weights.insert(0, d_l_w)

            # influence of previous layer: x_i dot with all w_i (each z_i = x_i * w.colum[i] + ...). Therefore d{z_i}_{x_i} is the colum{i} of w or w.T_i
            d_l_x = self.weights[layerIndex].T @ d_l_z

            #next layer output is this layer input
            d_l_f = d_l_x
        return d_weights, d_biases
    
    def apply_learning(self, d_weights, d_biases, scalar = 1):
        ...

    def act(self):
        availableActions = self.environment.availableActionsInEnv()
        x = np.array(self.environment.getRelativeEnv(self.index))
        all_x, all_z, y = self.forward(x)

        if random.random() < self.explorationRate:
            action = random.choice(availableActions)
        else:
            for a in availableActions:
                action[a] = 0
            action = np.argmax(y)
        return action
    
    def action_failed(self):
        print("how did this even happen?")

    def export_json(self, filePath):
        with open(filePath, "w") as f:
            json.dump({"NN": [
                            [
                                [j.tolist() for j in self.weights[i]], 
                                [j.tolist() for j in self.biases[i]]
                            ] for i in range(len(networkSizes)-1)
                            ]
                        }, f)

    def import_json(self, filePath):
        with open(filePath, "r") as f:
            pass

    def __str__(self):
        sep = "-"*20
        return f"\n{sep}\n".join([f"layer{i}=\nweight_shape={self.weights[i].shape}\nweight={self.weights[i].tolist()}\nbias_shape={self.biases[i].shape}\nbias={self.biases[i]}" for i in range(len(networkSizes)-1)])
            

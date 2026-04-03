import numpy as np
from Agent import Agent
import json
from const import *
from Environment import Environment
import math
import random
from typing import overload
"""
Goal
    Learn from playing only, think ahead 
Agent purpose
    Win Tic Tac Toe
    Earn maximum reward
Each action
    Train a game action upon final reward is realized

PATHWAY:
    layer f0(w0(x0)+b0) = x1 -> layer1  -> layer2 -> softmax -> loss(cross entropy)
"""

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid_x(x, providing_s=False):
    if providing_s:
        return x * (1-x)
    s = sigmoid(x)
    return s * (1-s)

def softmax(x, axis=-1):
    x = np.array(x)
    
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    
    return exp_x / sum_exp

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def d_relu(x, providing_s=False):
    """
    Derivative of ReLU.
    If providing_s is True, x is assumed to be the output of relu (s).
    Otherwise, x is the input to relu.
    """
    if providing_s:
        # x is s = relu(z), derivative is 1 if s > 0 else 0
        return (x > 0).astype(float)
    else:
        # x is z, derivative is 1 if z > 0 else 0
        return (x > 0).astype(float)

def identity(x):
    return x

def d_identity(x, providing_s=False):
    return np.ones_like(x)

def categorical_cross_entropy(better_s, s, epsilon=1e-15):
    s = np.clip(s, epsilon, 1. - epsilon)
    loss = -np.sum(better_s * np.log(s))
    return loss


func_to_d_func = {
    sigmoid: d_sigmoid_x,
    relu: d_relu,
    identity: d_identity,
}

chains = []
np.random.seed(None)

networkSizes = (BOARD_WIDTH * BOARD_HEIGHT, # input(environmentRelativeArray)
                32,
                32,
                BOARD_WIDTH * BOARD_HEIGHT)

neuralLayerCount = len(networkSizes) - 1 #not counting the input layer

# "To confuse the enemy, you must first confuse yourself" (Sun Tzu?)

class AgentNN(Agent):
    def __init__(self, index: int, environment : Environment):
        self.processing_functions = (
            relu,
            relu,
            identity,
        )
        self.index = index
        self.environment = environment
        self.learningRate = 0.01 #scales the amount to move in each input dimension
        self.depreciation = 0.75
        self.explorationRate = 0.5 #chaoticity
        self.exploration_decay = 0.99995
        super().__init__()
        self.weights = [
            np.random.randn(networkSizes[i+1], networkSizes[i]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.biases = [
            np.random.randn(networkSizes[i+1]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.history_forwards: list[tuple[list[np.ndarray], list[np.ndarray], int, np.ndarray]] = [] #history of forward values for back-prop
        # self.episodeActions = []

    def forward(self, x):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description

        :return: all_x, all_z, s
        
        run through the neural network 

        x = self.weights @ x + b
        is the same with rewriting an array x where each element is equal to w_x * x + b

        return all_x which is array of all input vectors of all layers and y (output) 
        return all_z which is all output after processing of all layers
        """
        all_x = [x]
        all_z = []
        for layerIndex in range(neuralLayerCount):
            z = self.weights[layerIndex] @ x + self.biases[layerIndex]
            all_z.append(z)
            f = self.processing_functions[layerIndex](z)
            x = f
            all_x.append(x)
        
        return all_x, all_z, x

    def clear_history(self):
        self.history_forwards = []

    def d_loss(self, s, better_s):
        return s - better_s #elementwise, s.i - better_s.i

    def backpropagate(self, all_x, all_z, s, better_s):
        """
        return Gradient to shape the network towards goal
        dl is change in loss, dl_[?] represent change in loss with respect to variable ?
        """
        # all_x, all_z, action = self.history_forward
        
        d_l_f = self.d_loss(s, better_s) # this is dl/df
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
            
            #reset d_l_[?] of previous layers
        # print("My bias: \n", d_biases)
        return d_weights, d_biases
    
    def apply_learning(self, d_weights, d_biases, scalar = 1):
        for layerIndex in range(neuralLayerCount):
            # print("Applying learning at layer", layerIndex, "\nBias:", self.biases[layerIndex].shape, "\nWeights:", self.weights[layerIndex].shape)
            # print("dBias:", d_biases[layerIndex].shape, "\ndWeights:", self.weights[layerIndex].shape)
            dw = d_weights[layerIndex] * self.learningRate * scalar
            db = d_biases[layerIndex] * self.learningRate * scalar
            # Clip to avoid extreme updates
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            self.weights[layerIndex] -= dw
            self.biases[layerIndex] -= db
    
    def one_hot(this, action):
        z = np.zeros(BOARD_WIDTH * BOARD_HEIGHT) 
        z[action] = 1
        return z

    def act(self):
        x = np.array(self.environment.getRelativeEnv(self.index))
        all_x, all_z, y = self.forward(x)

        #final processing
        availableActions = self.environment.availableActionsInEnv()

        masked_logits = y.copy()
        for i in range(len(y)):
            if i not in availableActions:
                masked_logits[i] = float('-inf')
        s = softmax(masked_logits)

        if random.random() < self.explorationRate:
            action: int = random.choice(availableActions)
        else:
            action: int = int(np.argmax(s))

        self.history_forwards.append((all_x, all_z, action, s))
        return action

    def action_failed(self):
        """
        Docstring for action_failed
        
        :param self: Description
        Punished for bad action immediately and remove invalid action from history 
        Current implementation: no punishment >_<
        """
        # all_x, all_z, action, s = self.history_forwards[-1] 
        # better_y = self.create_y_not_this(all_x[-1], action)
        # self.apply_learning(*self.backpropagate(all_x, all_z, better_y))
        self.history_forwards.pop(-1)

    def reward(self, reward):
        """
        Call when conclusive reward earned from game ending
        Reinforce if win or tie
        """

        if reward <= 0:
            return
            pass
        else:
            pass
        self.explorationRate = self.explorationRate * self.exploration_decay
        
        debug_losses = ""
        len_history = len(self.history_forwards)
        for history_index, history in enumerate(self.history_forwards):
            all_x, all_z, action, s = history
            # print([x.shape for x in all_x])
            better_s = self.one_hot(action)
            d_weights, d_biases = self.backpropagate(all_x, all_z, s, better_s)
            scalar = (reward) * (self.depreciation ** (len_history - history_index)) #reward ties also
            # print(f"Categorical Cross-Entropy Loss: {categorical_cross_entropy(better_s, s):.4f}")
            debug_losses += f" {categorical_cross_entropy(better_s, s):.4f}"
            self.apply_learning(d_weights, d_biases, scalar)

        self.clear_history()
        return debug_losses
    
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
            

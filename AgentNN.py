import numpy as np
from Agent import Agent
import json
from const import *
from Environment import Environment
import math
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

def d_sigmoid_x(x):
    s = sigmoid(x)
    return s * (1-s)


func_to_d_func = {
    sigmoid: d_sigmoid_x
}
# def softmax(x):
#     """Compute softmax values for each set of scores in x."""
#     # Subtract the maximum value for numerical stability
#     e_x = np.exp(x - np.max(x)) 
#     return e_x / e_x.sum(axis=0)

chains = []
np.random.seed(42)

networkSizes = (BOARD_WIDTH * BOARD_HEIGHT, # input(environmentRelativeArray)
                16,
                16,
                BOARD_WIDTH * BOARD_HEIGHT)

neuralLayerCount = len(networkSizes) - 1 #not counting the input layer

class AgentNN(Agent):
    

    def __init__(self, index: int, environment : Environment):
        self.processing_functions = (
            sigmoid,
            sigmoid,
            sigmoid,
        )
        self.index = index
        self.environment = environment
        self.learningRate = 0.01
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
    
    def convertInputToNpArray(self, action):
        pass
        # actionArray = np.zeros(BOARD_WIDTH * BOARD_HEIGHT)
        # actionArray[action] = 1
        # print(np.array(self.environment.getRelativeEnv(self.index)), actionArray)
        # return np.concatenate((np.array(self.environment.getRelativeEnv(self.index)), actionArray))

    def forward(self, x):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        
        run through the neural network 

        x = self.weights @ x + b
        is the same with rewriting an array x where each element is equal to w_x * x + b
        """
        all_x = [x]
        all_z = []
        for layerIndex in range(neuralLayerCount):
            x = self.weights[layerIndex] @ x + self.biases[layerIndex]
            all_z.append(x)
            x = self.processing_functions[layerIndex](x)
            all_x.append(x)
        return all_x, all_z, x

    def backpropagate(self, history_index, better_y):
        """
        return Gradient to shape the network towards goal
        d_l is change in loss, d_l_[?] represent change in loss with respect to variable ?
        """
        all_x, all_z, action = self.history_forwards[history_index]
        def loss(x, better_y):
            return (x - better_y) ** 2
        
        def d_loss(x):
            return 2 * (better_y - x)
        
        d_l_f = d_loss(all_x[-1]) # this is dl/df
        print("AgentNN: d_l_f \n", d_l_f)
        d_biases = []
        d_weights = []
        for layerIndex in range(neuralLayerCount -1, -1, -1):
            """
            f(wx+b)=f(z)=x1
            dl_z = dl_f * df_z : z input
            dl_x = dl_z * dz_x
            """
            z = all_z[layerIndex] # z of this layer
            d_l_z = d_l_f * func_to_d_func[self.processing_functions[layerIndex]](z)

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
        print("My bias: \n", d_biases)
        
        

        # normalize
        # total = sum(np.sum(i) for i in d_weights) + sum(np.sum(i) for i in d_biases)
        # d_weights = [i/total for i in d_weights]
        # d_biases = [i/total for i in d_biases]
        return d_weights, d_biases
    
    def apply_learning(self, d_weights, d_biases):
        for layerIndex in range(neuralLayerCount):
            # print("Applying learning at layer", layerIndex, "\nBias:", self.biases[layerIndex].shape, "\nWeights:", self.weights[layerIndex].shape)
            # print("dBias:", d_biases[layerIndex].shape, "\ndWeights:", self.weights[layerIndex].shape)
            self.weights[layerIndex] += d_weights[layerIndex] * self.learningRate
            self.biases[layerIndex] += d_biases[layerIndex] * self.learningRate

    def create_y_not_this(this, y, action):
        else_scale = 9/8
        this_scale = 7/8
        y1 = y.copy()
        y1 *= else_scale
        y1[action] *= this_scale / else_scale
        return y1

    def create_y_yes_this(this, y, action):
        else_scale = 7/8
        this_scale = 9/8
        y1 = y.copy()
        y1 *= else_scale
        y1[action] *= this_scale / else_scale
        return y1


    def act(self):
        # availableActions = self.environment.availableActionsInEnv()
        x = np.array(self.environment.getRelativeEnv(self.index))
        all_x, all_z, y = self.forward(x)
        action = np.argmax(y) #FIX: make it actual probability
        #add to learning memory
        print("AgentNN: probability distribution =\n", y)
        # self.episodeActions.append((x,y,action))
        self.history_forwards.append((all_x, all_z, action))
        return action
    
    def action_failed(self):
        """
        Docstring for action_failed
        
        :param self: Description
        Punished for bad action
        """
        # x, y, action = self.episodeActions[-1]
        all_x, all_z, action = self.history_forwards[-1] 
        better_y = self.create_y_not_this(all_x[-1], action)

        # backtrack, learn from invalid action
        print("AgentNN: My bias layer is", self.biases[-1])
        
        self.apply_learning(*self.backpropagate(-1, better_y))
        print("AgentNN: Illegal move made, refined probability=\n", better_y)
        print("AgentNN: My bias layer is refined\n", self.biases[-1])


    def relu(self, vector):
        return np.maximum(0,vector)

    def reward(self):
        pass
    
    def exportJson(self, filePath):
        with open(filePath, "w") as f:
            json.dump({"NN": [
                            [
                                [j.tolist() for j in self.weights[i]], 
                                [j.tolist() for j in self.biases[i]]
                            ] for i in range(len(networkSizes)-1)
                            ]
                        }, f)

    def importJson(self, filePath):
        with open(filePath, "r") as f:
            pass

    def __str__(self):
        sep = "-"*20
        return f"\n{sep}\n".join([f"layer{i}=\nweight_shape={self.weights[i].shape}\nweight={self.weights[i].tolist()}\nbias_shape={self.biases[i].shape}\nbias={self.biases[i]}" for i in range(len(networkSizes)-1)])
            

import numpy as np
from Agent import Agent
import json
from const import *
from Environment import Environment

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
chains = []
np.random.seed(42) 
 
networkSizes = (BOARD_WIDTH * BOARD_HEIGHT * 2, # input(environmentRelativeArray, actionArray)
                16,
                16,
                1)
neuralLayerCount = len(networkSizes) - 1 #not counting the input layer
class AgentNN(Agent):
    def __init__(self, index: int, environment : Environment):
        self.index = index
        self.environment = environment
        super().__init__()
        self.weights = [
            np.random.randn(networkSizes[i+1], networkSizes[i]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.biases = [
            np.random.randn(networkSizes[i+1]) * 0.01 
                for i in range(neuralLayerCount)
        ]
        self.episodeActions = []
    
    def convertInputToNpArray(self, action):
        actionArray = np.zeros(BOARD_WIDTH * BOARD_HEIGHT)
        actionArray[action] = 1
        print(np.array(self.environment.getRelativeEnv(self.index)), actionArray)
        return np.concatenate((np.array(self.environment.getRelativeEnv(self.index)), actionArray))

    def forward(self, x):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        
        run through the neural network 

        x = self.weights @ x + b
        is the same with rewriting an array x where each element is equal to w_x * x + b
        """
        for layerIndex in range(neuralLayerCount):
            x = self.weights[layerIndex] @ x + self.biases[layerIndex]
        return x

    def act(self):
        availableActions = self.environment.availableActionsInEnv()
        for action in availableActions:
            x = self.convertInputToNpArray(action)
            y = self.forward(x) # value of the action in this env
            print("action",action,"given rating of",y)

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
            

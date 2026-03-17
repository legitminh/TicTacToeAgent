from Agent import Agent
from const import *
import random
import json
from Environment import Environment
class AgentRL(Agent):
    def __init__(self, index : int, environment : Environment):
        self.environment = environment
        self.policy = {}
        self.play = []
        self.index = index
        self.explorationRate = 0.2 #chaoticity
        self.discountFactor = 0.75
        self.learningRate = 0.9 #descent speed
        self.exploration_decay = 0.999
        self.previous_env_map = None
        self.previous_action = None
    
    def get_relative_str_env(self):
        return self.environment.environmentListToFlattenString(self.environment.getRelativeEnv(self.index))

    def get_policy(self, envmap, action):
        if (envmap, action) not in self.policy:
            #initialize if not existing
            self.policy[(envmap, action)] = 0 + (random.random() - 0.5)**2
        return self.policy[(envmap, action)]
        
    def act(self):
        """
        choose an action based on the current environment
        """
        ...
    
    def teach_previous_action(self):
        """
        help the previous action
            the hope is the end state+action will give reward which the next time the end state+action is reached, the step leading to it will learn some
            Trickle down to the future!
        """
        ...

    
    def reward(self, reward):
        """
        when recieve reward, lean toward it
        """
        ...
    
    def lerp(self, a,b,fractionFromA):
        return a + (b-a) * fractionFromA
    
    def export_json(self, filePath):
        with open(filePath, "w") as f:
            json.dump({str(k): v for k, v in self.policy.items()}, f)

    def action_failed(self):
        pass

    def import_json(self, filePath):
        with open(filePath, "r") as f:
            raw = json.load(f)
            self.policy = {}
            for k, v in raw.items():
                # eval is dangerous; safer is literal_eval
                envmap, action = eval(k)
                self.policy[(envmap, action)] = v
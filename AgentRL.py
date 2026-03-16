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
        self.previousEnvironmentMap = None
        self.previousAction = None
    
    def getRelativeStrEnv(self):
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
        #TEMPLATE
        bestAction = None
        availableActions = self.environment.availableActionsInEnv()
        if len(availableActions)==0:
            pass
        elif random.random() < self.explorationRate:
            #exploration
            bestAction = random.choice(availableActions) 
        else:
            #exploitation
            maxValue = float('-inf')
            for a in availableActions:
                #choose the best action
                if self.get_policy(self.getRelativeStrEnv(), a) > maxValue:
                    maxValue = self.get_policy(self.getRelativeStrEnv(), a)
                    bestAction = a
        

        #Teaching phase
        self.teach_previous_action()

        self.previousEnvironmentMap = self.getRelativeStrEnv()
        self.previousAction = bestAction
        return bestAction
        #TEMPLATE
    
    def teach_previous_action(self):
        """
        help the previous action
            the hope is the end state+action will give reward which the next time the end state+action is reached, the step leading to it will learn some
            Trickle down to the future!
        """
        #TEMPLATE
        if (self.previousEnvironmentMap != None and self.previousAction  != None):
            currentQ = self.get_policy(self.previousEnvironmentMap, self.previousAction)
            # print("Debug teach_previous_action:",self.availableActionsInMap(self.environment.map))
            rewardActionList = [self.get_policy(self.getRelativeStrEnv(), action) for action in self.environment.availableActionsInEnv()]
            # target = sum( rewardActionList)/len(rewardActionList)
            target = max(rewardActionList) if len(rewardActionList)>0 else currentQ
            reward = 0 + self.discountFactor * target
            
            self.policy[(self.previousEnvironmentMap, self.previousAction)] = self.lerp(currentQ,reward , self.learningRate) 
        #TEMPLATE

    
    def reward(self, reward):
        """
        when recieve reward, lean toward it
        """
        #TEMPLATE
        self.explorationRate *= 0.99995
        if (self.previousEnvironmentMap  != None and self.previousAction != None):
            currentQ = self.get_policy(self.previousEnvironmentMap, self.previousAction)
            self.policy[(self.previousEnvironmentMap, self.previousAction)] = self.lerp(currentQ, reward , self.learningRate)
        #TEMPLATE
    
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
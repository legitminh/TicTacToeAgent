from abc import ABC, abstractmethod
from collections import defaultdict
import random
import json

BOARD_WIDTH = 3
BOARD_HEIGHT = 3

class Environment:
    def __init__(self):
        self.map = "_________"
    def reset(self):
        self.map = "_________"
    # consequence of the action done by player
    def impacted(self, action, player):
        self.editMap(action, str(player))
        # return self.checkWin()
    
    def editMap(self, index, value):
        self.map = self.map[:index] + value + self.map[index+1:]
    # return winner index or None
    def checkWin(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combinations:
            if self.map[combo[0]] == self.map[combo[1]] == self.map[combo[2]] != "_":
                return int(self.map[combo[0]])
        return None
    def __str__(self):
        return "\n".join([self.map[i*BOARD_WIDTH:(i+1)*BOARD_WIDTH] for i in range(BOARD_HEIGHT)])
        pass

"""
About learning:
    When an agent recieves a rewards, it will update its policy.
"""
class Agent(ABC):
    @property
    @abstractmethod
    def act(self):
        pass

class AgentRL(Agent):
    def __init__(self, index, environment):
        self.environment = environment
        self.policy = {} 
        self.play = []
        self.index = index
        self.explorationRate = 0.5
        self.discountFactor = 0.75
        self.learningRate = 0.9

        self.previousEnvironmentMap = None
        self.previousAction = None
    def getPolicy(self, envmap, action):
        #initialize if not existing
        if (envmap, action) not in self.policy:
            self.policy[(envmap, action)] = 0 + (random.random() - 0.5)**2
        return self.policy[(envmap, action)] 
        
    def availableActions(self):
        return self.availableActionsInMap(self.environment.map)
    def availableActionsInMap(self,map):
        actions = []
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if map[y*BOARD_WIDTH+x] == "_":
                    actions.append(y*BOARD_WIDTH+x)
        return actions 
    def act(self):
        bestAction = None
        availableActions = self.availableActions()
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
                if self.getPolicy(self.environment.map, a) > maxValue:
                    maxValue = self.getPolicy(self.environment.map, a)
                    bestAction = a
        

        #Teaching phase
        self.teachPreviousAction()

        self.previousEnvironmentMap = self.environment.map
        self.previousAction = bestAction
        return bestAction
    
    """
    help the previous action
        the hope is the end state+action will give reward which the next time the end state+action is reached, the step leading to it will learn some
        Trickle down to the future!!!
    """
    def teachPreviousAction(self):
        if (self.previousEnvironmentMap != None and self.previousAction  != None):
            currentQ = self.getPolicy(self.previousEnvironmentMap, self.previousAction)
            # print("Debug teachPreviousAction:",self.availableActionsInMap(self.environment.map))
            rewardActionList = [self.getPolicy(self.environment.map, action) for action in self.availableActionsInMap(self.environment.map)]
            # target = sum( rewardActionList)/len(rewardActionList)
            target = max(rewardActionList) if len(rewardActionList)>0 else currentQ
            reward = 0 + self.discountFactor * target
            
            self.policy[(self.previousEnvironmentMap, self.previousAction)] = self.lerp(currentQ,reward , self.learningRate) 

    # when recieve reward, lean toward it
    def reward(self, reward):
        self.explorationRate *= 0.9999

        if (self.previousEnvironmentMap  != None and self.previousAction  != None):
            currentQ = self.getPolicy(self.previousEnvironmentMap, self.previousAction)
            self.policy[(self.previousEnvironmentMap, self.previousAction)] = self.lerp(currentQ, reward , self.learningRate)
    
    def lerp(self, a,b,fractionFromA):
        return a + (b-a) * fractionFromA
    
    def exportJson(self, filePath):
        with open(filePath, "w") as f:
            json.dump({str(k): v for k, v in self.policy.items()}, f)

    def importJson(self, filePath):
        with open(filePath, "r") as f:
            raw = json.load(f)
            self.policy = {}
            for k, v in raw.items():
                # eval is dangerous; safer is literal_eval
                envmap, action = eval(k)  
                self.policy[(envmap, action)] = v

class AgentHuman(Agent):
    def __init__(self, index, environment):
        self.environment = environment
        self.index = index

        self.previousEnvironmentMap = None
        self.previousAction = None
    def act(self):
        action = int(input(f"Player {self.index} what is your move?: ").strip())
        return action

def trainAndExport():
    # create two agents
    env = Environment()
    singleAgent = AgentRL(0, env)
    # agents = [singleAgent, singleAgent]
    agents = []
    for i in range(2):
        agents.append(AgentRL(i, env))

    # accuracy measurement
    winCountByPlayer = [0] * 2
    winCounts = []
    EPISODES = 200000
    ROUND_REPORT_WINDOW = 10000
    for i in range(EPISODES):
        # make them one game
        env.reset()
        actionsIndex = random.randint(0,1)
        while env.checkWin() is None:
            actingAgent = agents[actionsIndex % 2]
            action = actingAgent.act()

            # no action available
            if action == None:
                break

            env.impacted(action, actingAgent.index)
            actionsIndex += 1
            if (i % ROUND_REPORT_WINDOW == 0):
                print(f"Agent {actingAgent.index} taking action {action}")
                print(f"Board state = {env.map}")
        
        # give rewards
        winner = env.checkWin()
        for agent in agents:
            if winner == agent.index:
                agent_reward = 1
            elif winner is None:
                agent_reward = 0
            else:
                agent_reward = -1
            agent.reward(agent_reward)

        # record result
        if winner != None:
            winCountByPlayer[winner] += 1
        winCounts.append(sum(winCountByPlayer))
        if (i % ROUND_REPORT_WINDOW == 0):
            print(f"episode {i} env = {env.map} winner = {winner}")

    # summary
    WINDOW_SIZE = 20
    for i in range(len(winCounts) - WINDOW_SIZE):
        if (i % WINDOW_SIZE == 0):
            print((winCounts[i+WINDOW_SIZE] - winCounts[i])/WINDOW_SIZE, end=" ")
    print("WinByPlayer:", "ties", winCountByPlayer,EPISODES - sum(winCountByPlayer))

    # export
    for i, agent in enumerate(agents):
        agent.exportJson(f"{agent.index}_AgentRL.json")

def aGameWithHuman(agent, humanIndex):
    env = Environment()
    # Play with human
    # make them one game
    human = AgentHuman(humanIndex, env)
    agent.environment = env
    env.reset()
    humanAndAI = [human, agent]
    actionsIndex = random.randint(0,1)
    while env.checkWin() is None:
        actingAgent = humanAndAI[actionsIndex % 2]
        action = actingAgent.act()

        # no action available
        if action == None:
            break

        env.impacted(action, actingAgent.index)
        actionsIndex += 1
        print(f"Agent {actingAgent.index} taking action {action}")
        print(f"{env}")
    winner = env.checkWin()
    print(f"env = {env.map} winner = {winner}")

if __name__ == "__main__":
    # trainAndExport()
    
    env = Environment()
    loadedAgent0 = AgentRL(0, env)
    loadedAgent0.importJson("0_AgentRL.json")
    #Enforce best behaviour
    loadedAgent0.explorationRate = 0
    aGameWithHuman(loadedAgent0, 1)
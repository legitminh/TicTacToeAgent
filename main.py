from collections import defaultdict
import random
import json
from Agent import Agent
from AgentNN import AgentNN 
from Environment import Environment
from const import *
from AgentRL import AgentRL

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
            # print("agent",actionsIndex,"action=", action)
            # no action available
            if action == None:
                break

            env.impacted(action, actingAgent.index)
            actionsIndex += 1
            if (i % ROUND_REPORT_WINDOW == 0):
                print(f"Agent {actingAgent.index} taking action {action}")
                print(f"Board state = {env.env}")
        
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
            print(f"episode {i} env = {env.toFlattenString()} winner = {winner}")

    # summary
    print("probability of non-ties (sub-optimal) sampling report")
    WINDOW_SIZE = 20
    for i in range(len(winCounts) - WINDOW_SIZE):
        if (i % WINDOW_SIZE == 0):
            print((winCounts[i+WINDOW_SIZE] - winCounts[i])/WINDOW_SIZE)
    print("WinByPlayer:", winCountByPlayer, "ties:" ,EPISODES - sum(winCountByPlayer))

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
    print(f"env = {env.env} winner = {winner}")

def playHumanAgentRL():
    env = Environment()
    loadedAgent0 = AgentRL(0, env)
    loadedAgent0.importJson("0_AgentRL.json")
    #Enforce best behaviour
    loadedAgent0.explorationRate = 0
    aGameWithHuman(loadedAgent0, 1)

# if __name__ == "__main__":
    # trainAndExport()
    # playHumanAgentRL()
    
    

if __name__ == "__main__":
    env = Environment()
    loadedAgent0 = AgentNN(0, env)
    # loadedAgent0.exportJson("0_AgentNN.json")
    print(loadedAgent0)
    aGameWithHuman(loadedAgent0, 1)
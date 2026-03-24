"""
Make AgentNN learn from AgentRL's policy
"""
import random

from AgentNN_backprop_only import AgentNN_simplified 
from AgentRL import AgentRL
from Environment import Environment
from utility import softmax, rd_array
import numpy as np

if __name__ == "__main__":
    env = Environment()
    loadedAgent0 = AgentRL(0, env)
    loadedAgent0.import_json("1_AgentRL.json")
    loadedAgent0.explorationRate = 0.0 #set to fully exploitative to use perfect policy

    loadedAgent1 = AgentNN_simplified(0, env)
    space_str_env = set()
    for p, value in loadedAgent0.policy.items():
        """
        Get values 
        """
        str_env, probability = p
        space_str_env.add(str_env)
    space_str_env = list(space_str_env)
    for i_episode in range(1000):
        for i, str_env in enumerate(random.sample(space_str_env, len(space_str_env)//2)):
            """
            Get optimal probability distribution from agentRL
            That is better_y for agentNN to learn from
            """
            env = Environment(str_env)
            available_actions = env.availableActionsInEnv()
            probabilities = []
            for action in range(env.getFlattenedSize()):
                if action not in available_actions:
                    probabilities.append(0)
                    continue
                probability = loadedAgent0.get_policy(str_env, action) #get probability distribution for action 0
                probabilities.append(probability)
            probabilities = softmax(probabilities)

            # print(env)
            # print(probabilities)

            loadedAgent1.environment = env
            all_x, all_z, y = loadedAgent1.forward(np.array(env.getRelativeEnv(loadedAgent1.index)))
            better_y = probabilities
            loadedAgent1.apply_learning(*loadedAgent1.backpropagate(all_x, all_z, better_y))
            # loadedAgent0.backpropagate()
            if (i==0) and i_episode % 10 == 0:
                print(loadedAgent1.loss(y, better_y))
                print(rd_array(y, 4))
                print(rd_array(better_y, 4))
        
    loadedAgent0.export_json("0_AgentNN.json")
    # print(loadedAgent0)
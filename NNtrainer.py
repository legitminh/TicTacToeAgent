from main import train
from Environment import Environment
from AgentRL import AgentRL
from AgentNN import AgentNN 

if __name__ == "__main__":
    env = Environment()
    loadedAgent0 = AgentRL(0, env)
    loadedAgent0.import_json("0_AgentRL.json")
    loadedAgent0.explorationRate = 0

    loadedAgent1 = AgentNN(1, env)
    train([loadedAgent0, loadedAgent1], 2000000)
    loadedAgent1.export_json("1_AgentNN.json")
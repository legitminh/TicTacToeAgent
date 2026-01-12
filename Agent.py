from abc import ABC, abstractmethod

"""
About learning:
    When an agent recieves a rewards, it will update its policy.
"""
class Agent(ABC):
    @property
    @abstractmethod
    def act(self):
        pass
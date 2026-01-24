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
    
    @abstractmethod
    def action_failed(self):
        """
        Docstring for action_failed
        
        :param self: Description
        Invalid action callback
        """
        pass
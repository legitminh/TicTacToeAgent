from const import *

class Environment:
    def __init__(self):
        self.reset()
    
    def availableActionsInEnv(self):
        actions = []
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.env[y*BOARD_WIDTH+x] == -1:
                    actions.append(y*BOARD_WIDTH+x)
        return actions

    def getRelativeEnv(self, observerIndex) -> list[int]:
        ans = []
        for i in self.env:
            ans.append(-1 if i == -1 else (int(i) - observerIndex) % 2) # 2 players 
        return ans


    def reset(self):
        """
        Docstring for reset
        
        :param self: Description
        env store -1 for empty cell, player indexes use 0,1,2
        """
        self.env = [-1] * self.getFlattenedSize()


    # consequence of the action done by player
    def impacted(self, action, player):
        self.editEnv(action, player)
        # return self.checkWin()
    
    def editEnv(self, index, value):
        self.env[index] = value
        # return winner index or None
    def checkWin(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combinations:
            if self.env[combo[0]] == self.env[combo[1]] == self.env[combo[2]] != -1:
                return self.env[combo[0]]
        return None
    
    def getFlattenedSize(self):
        return BOARD_WIDTH * BOARD_HEIGHT
    
    def environmentListToFlattenString(self, environmentList):
        a = ""
        for i in environmentList:
            a += "_" if i == -1 else str(i)
        return a
    
    def toFlattenString(self):
        return self.environmentListToFlattenString(self.env)
    
    def __str__(self):
        return "\n".join([self.toFlattenString()[i*BOARD_WIDTH:(i+1)*BOARD_WIDTH] for i in range(BOARD_HEIGHT)])
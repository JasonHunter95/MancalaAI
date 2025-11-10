import random
random.seed(109)

class Mancala:
    def __init__(self, pits_per_player=6, stones_per_pit = 4):
        """
        The constructor for the Mancala class defines several instance variables:

        pits_per_player: This variable stores the number of pits each player has.
        stones_per_pit: It represents the number of stones each pit contains at the start of any game.
        board: This data structure is responsible for managing the Mancala board.
        current_player: This variable takes the value 1 or 2, as it's a two-player game, indicating which player's turn it is.
        moves: This is a list used to store the moves made by each player. It's structured in the format (current_player, chosen_pit).
        p1_pits_index: A list containing two elements representing the start and end indices of player 1's pits in the board data structure.
        p2_pits_index: Similar to p1_pits_index, it contains the start and end indices for player 2's pits on the board.
        p1_mancala_index and p2_mancala_index: These variables hold the indices of the Mancala pits on the board for players 1 and 2, respectively.
        """
        self.pits_per_player = pits_per_player
        self.board = [stones_per_pit] * ((pits_per_player+1) * 2)  # Initialize each pit with stones_per_pit number of stones 
        self.players = 2
        self.current_player = 1
        self.moves = []
        self.p1_pits_index = [0, self.pits_per_player-1]
        self.p1_mancala_index = self.pits_per_player
        self.p2_pits_index = [self.pits_per_player+1, len(self.board)-1-1]
        self.p2_mancala_index = len(self.board)-1
        
        # Zeroing the Mancala for both players
        self.board[self.p1_mancala_index] = 0
        self.board[self.p2_mancala_index] = 0

    def display_board(self):
        """
        Displays the board in a user-friendly format
        """
        player_1_pits = self.board[self.p1_pits_index[0]: self.p1_pits_index[1]+1]
        player_1_mancala = self.board[self.p1_mancala_index]
        player_2_pits = self.board[self.p2_pits_index[0]: self.p2_pits_index[1]+1]
        player_2_mancala = self.board[self.p2_mancala_index]

        print('P1               P2')
        print('     ____{}____     '.format(player_2_mancala))
        for i in range(self.pits_per_player):
            if i == self.pits_per_player - 1:
                print('{} -> |_{}_|_{}_| <- {}'.format(i+1, player_1_pits[i], 
                        player_2_pits[-(i+1)], self.pits_per_player - i))
            else:    
                print('{} -> | {} | {} | <- {}'.format(i+1, player_1_pits[i], 
                        player_2_pits[-(i+1)], self.pits_per_player - i))
            
        print('         {}         '.format(player_1_mancala))
        turn = 'P1' if self.current_player == 1 else 'P2'
        print('Turn: ' + turn)
        
    def valid_move(self, pit):
        """
        Function to check if the pit chosen by the current_player is a valid move.
        if the pit chosen is within the range of pits and the pit is not empty.
        """
        if not (1 <= pit <= self.pits_per_player):
            return False
        start = self.p1_pits_index[0] if self.current_player == 1 else self.p2_pits_index[0]
        pit_index = start + (pit - 1)
        return self.board[pit_index] > 0
        
    def random_move_generator(self):
        """
        Function to generate random valid moves with non-empty pits for the random player
        Returns a random pit number in the valid range that is not empty
        """
        # write your code here
        start = self.p1_pits_index[0] if self.current_player == 1 else self.p2_pits_index[0]
        choices = [i - start + 1 
                   for i in range(start, start + self.pits_per_player)
                   if self.board[i] > 0] # list comprehension for getting all non-empty pits
        if choices:
            return random.choice(choices)
        else:
            return None
    
    def is_on_players_side(self, j, player_pits_index):
            """Returns a bool that tells if the given pit is on the current players side"""
            start, end = player_pits_index
            return start <= j <= end
            

    def opposite(self, j):
            """Calculates and returns the index of the pit 
            that is directly opposite to pit j so the capture rule can be applied"""
            return 2 * self.pits_per_player - j
            
    def distribute_stones(self, pit, player_pits_index, player_mancala_index):
        # print(len(self.board))
        """Handles all stone distribution logic"""
        board_range = len(self.board)
        opponent_mancala_index = (self.p2_mancala_index if player_mancala_index == self.p1_mancala_index 
                                  else self.p1_mancala_index)
            
        stones = self.board[pit]
        if stones == 0:
            return
        self.board[pit] = 0
        
        i = pit
        prev = pit
        while stones > 0:
            i = (i + 1) % board_range
            if i == opponent_mancala_index:
                continue
            self.board[i] += 1
            stones -= 1
            prev = i          
                


        if self.is_on_players_side(prev, player_pits_index) and self.board[prev] == 1:
            opp = self.opposite(prev)
            if 0 <= opp < board_range and self.board[opp] > 0:
                captured = self.board[opp] + 1
                self.board[opp] = 0
                self.board[prev] = 0
                self.board[player_mancala_index] += captured
        return
    
    
    def play(self, pit):
        """
        This function simulates a single move made by a specific player using their selected pit. It primarily performs three tasks:
        1. It checks if the chosen pit is a valid move for the current player. If not, it prints "INVALID MOVE" and takes no action.
        2. It verifies if the game board has already reached a winning state. If so, it prints "GAME OVER" and takes no further action.
        3. After passing the above two checks, it proceeds to distribute the stones according to the specified Mancala rules.

        Finally, the function then switches the current player, allowing the other player to take their turn.
        """
        # commented this out because its overwhelming the jupyter nb output with turns
        print(f"Player {self.current_player} chose pit: {pit}")
        
        if self.winning_eval() == True:
            print("GAME OVER")
            return self.board
        
        if self.valid_move(pit) == False:
            print("INVALID MOVE")
            return self.board
        
        self.moves.append((self.current_player, pit))
        
        if self.current_player == 1:
            idx = self.p1_pits_index[0] + (pit - 1)
            self.distribute_stones(idx, self.p1_pits_index, self.p1_mancala_index)
        
        else:
            idx = self.p2_pits_index[0] + (pit - 1)
            self.distribute_stones(idx, self.p2_pits_index, self.p2_mancala_index)
            
        # helper for checking side sums
        def side_sum(lo, hi):
            """
            Calculates and returns the sum of stones on a single player's side of the board.
            """
            return sum(self.board[lo:hi + 1])

        if side_sum(self.p1_pits_index[0], self.p1_pits_index[1]) == 0 or side_sum(self.p2_pits_index[0], self.p2_pits_index[1]) == 0:
            # sweep both sides into respective mancalas
            for i in range(self.p1_pits_index[0], self.p1_pits_index[1] + 1):
                self.board[self.p1_mancala_index] += self.board[i]
                self.board[i] = 0
            for i in range(self.p2_pits_index[0], self.p2_pits_index[1] + 1):
                self.board[self.p2_mancala_index] += self.board[i]
                self.board[i] = 0
                
        # switch player if the game isn't over yet
        if not self.winning_eval():
            self.current_player = 2 if self.current_player == 1 else 1

        return self.board


    def winning_eval(self):
        """
        Function to verify if the game board has reached the winning state.
        Hint: If either of the players' pits are all empty, then it is considered a winning state.
        I grab the sum of stones in both players pits and if either side is zero, the game is over.
        """
        # write your code here
        p1_empty = sum(self.board[self.p1_pits_index[0]:self.p1_pits_index[1] + 1]) == 0
        p2_empty = sum(self.board[self.p2_pits_index[0]:self.p2_pits_index[1] + 1]) == 0
        return p1_empty or p2_empty

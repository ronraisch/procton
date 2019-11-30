import numpy as np


class GameState:
    # whites are positive in our format (blace are negatives)
    # player_turn=1 => white plays
    # the positive rotation in the board is in the white's direction
    # the first index of eating places are for whites
    # move="out" => removing soldier from game
    INITIAL_SOLDIER = 0
    NUMBER_OF_POSITIONS = 26 + INITIAL_SOLDIER
    NUMBER_OF_SOLDIERS = 15
    HOME_SIZE = 6
    MIN_DICE_VALUE = 1
    MAX_DICE_VALUE = 6
    KILLED_SOLDIERS_INDEX = NUMBER_OF_POSITIONS + INITIAL_SOLDIER
    DICE_RESULT = KILLED_SOLDIERS_INDEX + 2

    def __init__(self, l):
        self.state = l

    def game_over(self):
        return self.state[GameState.NUMBER_OF_SOLDIERS - 1] == GameState.NUMBER_OF_SOLDIERS or \
               self.state[0] == GameState.NUMBER_OF_SOLDIERS

    @staticmethod
    def check_endgame(state, player_turn):
        dummie = (GameState.NUMBER_OF_POSITIONS - GameState.HOME_SIZE - 2) / 2
        first_index = dummie * (player_turn + 1) + 1
        sum = state[(GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) / 2]
        for i in range(first_index, first_index + GameState.HOME_SIZE):
            if state[i] * player_turn > 0:
                sum += state[i]
        return abs(sum) == GameState.NUMBER_OF_SOLDIERS

    @staticmethod
    def find_highest_position(state, player_turn):
        index = state[(GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) / 2]
        highest_position = index - player_turn
        for i in range(index, index - player_turn * GameState.HOME_SIZE, -player_turn):
            if state[i] * player_turn > 1:
                highest_position = i
        return highest_position

    @staticmethod
    def check_endendgame(state, player_turn):
        highest_position = GameState.find_highest_position(state, player_turn)
        # 24->1
        # 19->6
        # 1->1
        # 6->6
        normalized_hp = (GameState.NUMBER_OF_POSITIONS - 1 - highest_position) * (
                player_turn + 1) / 2 - highest_position * (player_turn - 1) / 2
        if state[GameState.DICE_RESULT] > normalized_hp and state[
            GameState.DICE_RESULT + 1] > normalized_hp:
            print(normalized_hp)
            return True and GameState.check_endgame(state, player_turn)
        return False

    def get_possible_moves(self, player_turn):
        possible_moves = []
        dice_result = self.state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
        for i in range(GameState.NUMBER_OF_POSITIONS):
            if GameState.is_legal_move(self.state, [i, i + dice_result[0]], player_turn):
                move1 = [i, i + dice_result[0]]
                eval_state = GameState.make_move_on_state([move1], self.state, player_turn)
            else:
                continue
            for j in range(GameState.NUMBER_OF_POSITIONS):
                if GameState.is_legal_move(eval_state, [j, j + dice_result[1]], player_turn):
                    possible_moves.append([move1, [j, j + dice_result[1]]])
        return possible_moves

    @staticmethod
    def is_double(state):
        dice_result = state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
        return dice_result[0] == dice_result[1]

    def get_possible_states_double(self, player_turn):
        possible_moves = self.get_possible_moves(player_turn)
        possible_states = []
        for move in possible_moves:
            possible_states.append(GameState.make_move_on_state(move, self.state, player_turn))
        new_possible_states = []
        for state in possible_states:
            tmp_state = GameState(state)
            possible_moves = tmp_state.get_possible_moves(player_turn)
            tmp_states = []
            for move in possible_moves:
                tmp_states.append(GameState.make_move_on_state(move, tmp_state.state, player_turn))
            new_possible_states += tmp_states
        if (len(new_possible_states)) > 1:
            return np.unique(new_possible_states, axis=0)
        return new_possible_states

    def get_possible_states(self, player_turn):
        if not GameState.is_double(self.state):
            possible_moves = self.get_possible_moves(player_turn)
            possible_states = []
            for move in possible_moves:
                possible_states.append(GameState.make_move_on_state(move, self.state, player_turn))
            return possible_states
        return self.get_possible_states_double(player_turn)

    @staticmethod
    def is_legal_move(state, move, player_turn):
        move = [int(move[0]), int(move[1])]
        # if move[0] is in the same color of the player.
        if state[move[0]] * player_turn <= 0:
            return False
        dice_result = state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]

        # if move corresponds to dice results
        # TODO: check if condition necessary (for optimization)
        if (move[1] - move[0]) * player_turn not in dice_result:
            return False

        # if move in the board
        if move[1] >= GameState.NUMBER_OF_POSITIONS:
            return False
            # return GameState.check_endendgame(state, player_turn) and GameState.end_game_mode()

        # if state[move[1]] is available
        if state[move[1]] * player_turn < -1:
            return False

        if move[1] == (GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) / 2 and not GameState.check_endgame(state, player_turn):
            return False
        return True

    @staticmethod
    def make_move_on_state(moves, state, player_turn):
        tmp_state = state.copy()
        for move in moves:
            move = [int(move[0]), int(move[1])]
            # making the move
            tmp_state[move[0]] -= player_turn
            tmp_state[move[1]] += player_turn
            # handling with eating concept
            if tmp_state[move[1]] == 0:
                tmp_state[move[1]] += player_turn
                tmp_state[GameState.KILLED_SOLDIERS_INDEX + (player_turn + 1) / 2] += 1
        return tmp_state

    def evaluate(self):
        score = np.arange(-GameState.NUMBER_OF_POSITIONS / 2 + 1, GameState.NUMBER_OF_POSITIONS / 2)
        return np.dot(score, self.state[1:GameState.NUMBER_OF_POSITIONS])

# # add eating and removing soldiers
#         possible_moves = []
#         dice_result = self.state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
#         endgame = False
#         # solve double situation
#         if dice_result[0] == dice_result[1]:
#             dice_result += dice_result
#         for result in dice_result:
#             move = []
#             tmp_state = self.state
#             # looping through all positions on board
#             for i in range(GameState.INITIAL_SOLDIER, GameState.NUMBER_OF_POSITIONS):
#                 # checking if this spot on the board is occupied by our player
#                 if player_turn * self.state[i] > 0:
#                     next_index = int(i + player_turn * result)
#                 # checking if there are opponent's soldiers on the dices' positions
#                     if (player_turn * tmp_state[next_index] >= 0):
#                         tmp_state = GameState.make_move_on_state([[i, next_index]], tmp_state, player_turn)
#                         move.append([i, next_index])
#             possible_moves.append(move)
#         return possible_moves

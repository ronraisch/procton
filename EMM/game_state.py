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
    score1 = np.power(np.arange(-NUMBER_OF_POSITIONS / 2 + 1, 0), 3)
    score2 = np.power(np.arange(1, NUMBER_OF_POSITIONS / 2), 3)
    SCORE = np.concatenate((score1, -score2))
    KILLED_SOLDIER = "killed"

    def __init__(self, l):
        self.state = l

    def game_over(self):
        return self.state[GameState.NUMBER_OF_POSITIONS - 1] == GameState.NUMBER_OF_SOLDIERS or \
               self.state[0] == GameState.NUMBER_OF_SOLDIERS

    @staticmethod
    def is_double(state):
        dice_result = state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
        return dice_result[0] == dice_result[1]

    @staticmethod
    def check_endgame(state, player_turn):
        home_index = (GameState.NUMBER_OF_POSITIONS - 1) * (player_turn + 1) // 2
        # first_index = int(dummie * (player_turn + 1) + 1)
        sum = 0
        sum += state[home_index]
        for i in range(home_index - player_turn, home_index - player_turn - player_turn * GameState.HOME_SIZE,
                       -player_turn):
            if state[i] * player_turn > 0:
                sum += state[i]
        return abs(sum) == GameState.NUMBER_OF_SOLDIERS

    @staticmethod
    def find_highest_position(state, player_turn):
        home_index = (GameState.NUMBER_OF_POSITIONS - 1) * (player_turn + 1) // 2
        highest_position = home_index - player_turn
        for i in range(highest_position, highest_position - player_turn * GameState.HOME_SIZE, -player_turn):
            if state[i] * player_turn >= 1:
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
        if state[GameState.DICE_RESULT] >= normalized_hp and state[GameState.DICE_RESULT + 1] >= normalized_hp:
            return True and GameState.check_endgame(state, player_turn)
        return False

    @staticmethod
    def check_eaten(state, player_turn):
        return state[GameState.KILLED_SOLDIERS_INDEX + (-player_turn + 1) // 2] > 0

    def get_move_iterator(self, player_turn, dice_result):
        moves = []
        # 1->24
        # ....6->19 for player_turn=1
        # dice->dice for player_turn=-1
        dice_pos = []
        for dice_val in dice_result:
            dice_white = GameState.NUMBER_OF_POSITIONS - 1 - dice_val
            dice_pos.append(int((1 + player_turn) / 2 * dice_val + (1 - player_turn) / 2 * dice_white))

        # has eaten checkers
        if GameState.check_eaten(self.state, player_turn):
            for i in range(len(dice_result)):
                moves.append(([dice_pos[i], GameState.KILLED_SOLDIER], dice_result[i]))

        # regular mode
        else:
            for i in range(1, GameState.NUMBER_OF_POSITIONS - 1):
                for dice_val in dice_result:
                    moves.append(([i, i + dice_val * player_turn], dice_val))
        return moves

    def get_possible_states(self, player_turn):
        if not self.is_double(self.state):
            return self.get_possible_states_not_double(player_turn)
        return self.get_possible_states_double(player_turn)

    def get_possible_states_not_double(self, player_turn):
        possible_states = []
        one_dice_states = []
        dice_result = list(self.state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2])
        moves1 = self.get_move_iterator(player_turn, dice_result)
        for move1 in moves1:
            if GameState.is_legal_move(self.state, move1[0], player_turn):

                eval_state1 = GameState.make_move_on_state([move1[0]], self.state, player_turn)
                eval_game_state = GameState(eval_state1)
            else:
                continue
            current_dice = list(dice_result)
            current_dice.remove(move1[1])
            moves2 = eval_game_state.get_move_iterator(player_turn, current_dice)
            states = []
            for move in moves2:
                if GameState.is_legal_move(eval_state1, move[0], player_turn):
                    eval_state2 = GameState.make_move_on_state([move[0]], eval_state1, player_turn)
                    states.append(eval_state2)
            if len(states) == 0:
                one_dice_states.append(eval_state1)
            else:
                possible_states += states
        if len(possible_states) == 0:
            possible_states += one_dice_states
        if len(possible_states) == 0:
            possible_states.append(self.state)
        return list(np.unique(possible_states, axis=0))

    def get_possible_states_double(self, player_turn):
        possible_states = self.get_possible_states_not_double(player_turn)
        new_possible_states = []
        for state in possible_states:
            tmp_state = GameState(state)
            new_possible_states += tmp_state.get_possible_states_not_double(player_turn)
        if (len(new_possible_states)) > 1:
            return np.unique(new_possible_states, axis=0)
        if len(new_possible_states) == 0:
            return [self.state]
        return new_possible_states

    @staticmethod
    def get_free_spots(state, player_turn):
        # get the initial index for searching
        index = (GameState.NUMBER_OF_POSITIONS - 2) * (-player_turn + 1) / 2 + (player_turn + 1) / 2
        free_sposts = []
        # searching for spots with player_turn type or empty or with one enemy soldier
        for i in range(index, index + player_turn * GameState.HOME_SIZE, player_turn):
            if state[i] * player_turn >= -1:
                free_sposts.append(i)
        return free_sposts

    @staticmethod
    def is_legal_move_eaten(state, move, player_turn):
        if state[int(move[0])] * player_turn >= -1:
            return True
        return False

    @staticmethod
    def is_legal_move(state, move, player_turn):
        # handling with soldier coming back from the dead
        if GameState.check_eaten(state, player_turn):
            return GameState.is_legal_move_eaten(state, move, player_turn)

        src = int(move[0])
        dest = int(move[1])
        # if move[0] is in the same color of the player.
        if state[src] * player_turn <= 0:
            return False

        # trying to remove soldiers before endgame
        if dest == (GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) / 2 and not GameState.check_endgame(state, player_turn):
            return False

        # if move in the board
        if dest > GameState.NUMBER_OF_POSITIONS - 1 or dest < 0:
            if not GameState.check_endendgame(state, player_turn):
                return False
            if src != GameState.find_highest_position(state, player_turn):
                return False
            move[1] = (GameState.NUMBER_OF_POSITIONS - 1) * (player_turn + 1) // 2
            dest = move[1]
        # if state[move[1]] is available
        if state[dest] * player_turn < -1:
            return False

        return True

    @staticmethod
    def make_move_on_state(moves, state, player_turn):
        tmp_state = state.copy()
        if len(moves) == 0:
            return tmp_state
        for move in moves:
            # bringing back eaten soldier
            if move[1] == GameState.KILLED_SOLDIER:
                move[0] = int(move[0])
                tmp_state[move[0]] += player_turn
                tmp_state[GameState.KILLED_SOLDIERS_INDEX + (-player_turn + 1) // 2] -= 1
                # for the case where I eat with my eaten soldier
                move[1] = move[0]
            else:
                # fixing int problems
                move = [int(move[0]), int(move[1])]
                # making the normal move
                tmp_state[move[0]] -= player_turn
                tmp_state[move[1]] += player_turn
            # handling with eating concept
            if tmp_state[move[1]] == 0:
                tmp_state[move[1]] += player_turn
                tmp_state[GameState.KILLED_SOLDIERS_INDEX + (player_turn + 1) // 2] += 1

        return tmp_state

    def get_open_houses(self, player_turn):
        total = 0
        # counting each open house => |state|=1
        for i in range(1, GameState.NUMBER_OF_POSITIONS - 1):
            if self.state[i] * player_turn == 1:
                total += i * (player_turn + 1) // 2 + (25 - i) * (player_turn - 1) // -2
        return total

    def get_eaten(self, player_turn):
        # number of killed soldiers
        return self.state[GameState.KILLED_SOLDIERS_INDEX + (-player_turn + 1) // 2]

    def value_of_houses(self, player_turn):
        total = 0
        for pos in self.state[1:GameState.NUMBER_OF_POSITIONS - 1]:
            if pos * player_turn > 1:
                total += GameState.value_of_house(pos * player_turn)
        return total

    @staticmethod
    def value_of_house(size):
        for i in range(2, 6):
            if size == i:
                return (13 / np.pi) * (np.arctan(2.5 - 0.75 * i) + np.pi / 2)

        return (13 / np.pi) * (np.arctan(2.5 - 0.75 * 6) + np.pi / 2)

    def get_soldiers_positions(self):
        # returning just the board position part from the state vector
        return self.state[1:GameState.NUMBER_OF_POSITIONS - 1]

    # Evaluation function for board state
    def evaluate(self, player_turn):
        return GameState.value_of_houses(self, player_turn) - 1.7 ** GameState.get_open_houses(self, player_turn) + \
               100 ** (GameState.get_eaten(self, -player_turn) - GameState.get_eaten(self, player_turn))

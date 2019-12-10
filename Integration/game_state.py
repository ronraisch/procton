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
        return self.state[GameState.NUMBER_OF_SOLDIERS - 1] == GameState.NUMBER_OF_SOLDIERS or \
               self.state[0] == GameState.NUMBER_OF_SOLDIERS

    @staticmethod
    def is_double(state):
        dice_result = state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
        return dice_result[0] == dice_result[1]

    @staticmethod
    def check_endgame(state, player_turn):
        dummie = (GameState.NUMBER_OF_POSITIONS - GameState.HOME_SIZE - 2) // 2
        first_index = int(dummie * (player_turn + 1) + 1)
        sum = state[(GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) // 2]
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

    @staticmethod
    def check_eaten(state, player_turn):
        return state[GameState.KILLED_SOLDIERS_INDEX + (-player_turn + 1) // 2] > 0

    def get_move_iterator(self, player_turn, dice_val):
        moves = []
        # 1->24
        # ....6->19 for player_turn=1
        # dice->dice for player_turn=-1
        dice_white = GameState.NUMBER_OF_POSITIONS - 1 - dice_val
        dice_pos = int((1 + player_turn) / 2 * dice_val + (1 - player_turn) / 2 * dice_white)
        if GameState.check_eaten(self.state, player_turn):
            return [[dice_pos, GameState.KILLED_SOLDIER]]
        if GameState.check_endgame(self.state, player_turn):
            if self.state[dice_pos] * player_turn > 0:
                moves.append(
                    [dice_pos, (GameState.NUMBER_OF_POSITIONS - 1) * (1 + player_turn) / 2])
            first_index = int((1 - player_turn) / 2 + (1 + player_turn) / 2 * (
                    GameState.NUMBER_OF_POSITIONS - 2))
            for i in range(first_index, first_index - player_turn * GameState.HOME_SIZE,
                           -player_turn):
                if GameState.NUMBER_OF_POSITIONS - 1 > i - dice_val * player_turn > 0 and \
                        self.state[i] * player_turn > 0:
                    moves.append([i, i - dice_val * player_turn])
            if len(moves) == 0:
                moves.append([GameState.find_highest_position(self.state, player_turn),
                              (GameState.NUMBER_OF_POSITIONS - 1) * (1 + player_turn) / 2])
        else:
            for i in range(GameState.NUMBER_OF_POSITIONS):
                moves.append([i, i + dice_val * player_turn])
        return moves

    def get_possible_states(self, player_turn):
        if not self.is_double(self.state):
            return self.get_possible_states_not_double(player_turn)
        return self.get_possible_states_double(player_turn)

    def get_possible_states_not_double(self, player_turn):
        board = self.state.copy()
        board[GameState.DICE_RESULT:GameState.DICE_RESULT + 2] = [
            board[GameState.DICE_RESULT + 1], board[GameState.DICE_RESULT]]
        other_dices = GameState(board)
        possible_states = self.get_possible_states_wo_eaten(player_turn)
        other_states = other_dices.get_possible_states_wo_eaten(player_turn)
        for state in other_states:
            state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2] = \
                [state[GameState.DICE_RESULT+1], board[GameState.DICE_RESULT]]
        possible_states += other_states
        possible_states = [list(a) for a in possible_states]
        if list(self.state) in possible_states:
            possible_states.remove(list(self.state))
        possible_states = [np.array(x) for x in possible_states]
        return [self.state] if len(possible_states) == 0 else list(np.unique(possible_states, axis=0))

    def get_possible_states_wo_eaten(self, player_turn):
        possible_states = []
        dice_result = self.state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2]
        moves1 = self.get_move_iterator(player_turn, dice_result[0])
        counter = 0
        for move1 in moves1:
            eval_state1 = None
            if GameState.is_legal_move(self.state, move1, player_turn):
                eval_state1 = GameState.make_move_on_state([move1], self.state, player_turn)
            else:
                counter += 1
                if counter < len(moves1):
                    continue
            if eval_state1 is not None:
                tmp = GameState(eval_state1)
            else:
                tmp = GameState(self.state.copy())
            moves2 = tmp.get_move_iterator(player_turn, dice_result[1])
            states = []
            for move in moves2:
                if GameState.is_legal_move(tmp.state, move, player_turn):
                    eval_state2 = GameState.make_move_on_state([move], tmp.state, player_turn)
                    states.append(eval_state2)
            if len(states) == 0:
                possible_states.append(tmp.state)
            else:
                possible_states += states
        return possible_states

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
        if move[1] > GameState.NUMBER_OF_POSITIONS - 2 or move[1] < 1:
            return False
            # return GameState.check_endendgame(state, player_turn) and GameState.end_game_mode()

        # if state[move[1]] is available
        if state[move[1]] * player_turn < -1:
            return False
        # trying to remove soldiers before endgame
        if move[1] == (GameState.NUMBER_OF_POSITIONS - 1) * (
                player_turn + 1) / 2 and not GameState.check_endgame(state, player_turn):
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
        return total//2

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
                return 2 ** (7 - i)
        return 2


    def get_soldiers_positions(self):
        # returning just the board position part from the state vector
        return self.state[1:GameState.NUMBER_OF_POSITIONS - 1]

    # Evaluation function for board state
    def evaluate(self, player_turn):
        return GameState.value_of_houses(self, player_turn) - 2 ** GameState.get_open_houses(self, player_turn) + \
               100 ** (GameState.get_eaten(self, -player_turn) - GameState.get_eaten(self, player_turn))

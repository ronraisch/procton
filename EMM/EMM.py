# import numpy as np
# from game_state import GameState
from GUI import *


def set_dices(state):
    r = np.random.randint(1, 7, 2)
    x=[]
    if type(state)==type(x):
        state=np.array(state)
    state[GameState.DICE_RESULT:GameState.DICE_RESULT + 2] = r



def get_prob(state):
    if state[GameState.DICE_RESULT] == state[GameState.DICE_RESULT + 1]:
        return 1 / 36
    return 1 / 18


counter = 0


def EMM(state_list, depth, player_turn):
    state = GameState(state_list)
    max_val = -np.inf
    max_state = None
    for board in state.get_possible_states(player_turn):
        result = EMM_wo_dices(board, depth - 1, -np.inf, -np.inf, -player_turn)
        if result[0] > max_val:
            max_val = result[0]
            max_state = board
    return max_val, max_state


def EMM_wo_dices(state_list, depth, alpha, beta, player_turn):
    # global counter
    # counter += 1
    # print(counter)
    state = GameState(state_list)
    if depth == 0 or state.game_over():
        return state.evaluate(player_turn), state_list

    max_val = -np.inf
    max_state = None
    for i in range(GameState.MIN_DICE_VALUE, GameState.MAX_DICE_VALUE + 1):
        for j in range(i, GameState.MAX_DICE_VALUE + 1):
            new_board = state_list.copy()
            new_board[GameState.DICE_RESULT:GameState.DICE_RESULT + 2] = np.array([i, j])
            state_dice = GameState(new_board)
            for board in state_dice.get_possible_states(player_turn):
                tmp = GameState(board)
                val, dummie = EMM_wo_dices(tmp.state, depth - 1, alpha, beta, -player_turn)
                val *= get_prob(tmp.state)

                if max_val < val:
                    max_val = val
                    max_state = tmp.state.copy()
                alpha = max(alpha, val)
                # if beta <= alpha:
                #     print(max_val, max_state, "alpha beta")
                #     return max_val, max_state
    # print(max_val, max_state)
    return max_val, max_state


# result = EMM(INITIAL_STATE, 1, 1)
# print(result)
# GUI_state(result[1])
#
# my_state = GameState(np.array(result[1]))
# print(my_state.evaluate())
# GUI_state(INITIAL_STATE)
np.random.seed(657)
board = INITIAL_STATE
p = 1
for i in range(20):
    print(board)
    GUI_state(board)
    result = EMM(board, 1, p)
    board = result[1]
    set_dices(board)
    p *= -1


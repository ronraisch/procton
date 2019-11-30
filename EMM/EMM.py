# import numpy as np
# from game_state import GameState
from GUI import *


def get_prob(state):
    if state[GameState.DICE_RESULT] == state[GameState.DICE_RESULT]:
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
        return state.evaluate(), state_list

    max_val = -np.inf
    max_state = None
    for i in range(GameState.MIN_DICE_VALUE, GameState.MAX_DICE_VALUE+1):
        for j in range(i, GameState.MAX_DICE_VALUE+1):
            new_board = state_list.copy()
            new_board[GameState.DICE_RESULT:GameState.DICE_RESULT + 2] = np.array([i, j])
            state_dice = GameState(new_board)
            # GUI_state(state_dice.state)
            print(state_dice.state,player_turn)
            for board in state_dice.get_possible_states(player_turn):
                print("Sdf")
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



result = EMM(INITIAL_STATE, 2, 1)
print(result)
GUI_state(result[1])

my_state = GameState(np.array(result[1]))
print(my_state.evaluate())

# if depth == 0 or state.game_over():
#     return evaluate(state)
# game = GameState(state)
# if player_turn == 1:
#     max_eval = -np.inf
#     for position in game.get_possible_states(1):
#         eval = p(position) * p(state) * EMM(position, depth - 1, -1)
#         max_eval = np.max(max_eval, eval)
#     return max_eval
# else:
#     min_eval = np.inf
#     for position in game.get_possible_states(-1):
#         eval = p(position) * p(state) * EMM(position, depth - 1, 1)
#         min_eval = np.max(min_eval, eval)
#     return min_eval

# import numpy as np
# from game_state import GameState
from GUI import *


def p(state):
    if state[GameState.DICE_RESULT] == state[GameState.DICE_RESULT]:
        return 1 / 36
    return 1 / 18


def EMM(state_list, depth, alpha, beta, player_turn):
    state = GameState(state_list)
    if depth == 0 or state.game_over():
        return state.evaluate(),state_list

    max_val = -np.inf
    max_state = None
    for board in state.get_possible_states(player_turn):
        for i in range(GameState.MIN_DICE_VALUE,GameState.MAX_DICE_VALUE):
            for j in range(i,GameState.MAX_DICE_VALUE):
                new_board=board.copy()
                new_board[GameState.DICE_RESULT:GameState.DICE_RESULT+2]=np.array([i,j])
                tmp = GameState(new_board)
                val, dummie = EMM(tmp.state, depth - 1, alpha, beta, -player_turn) * p(tmp.state)
                if max_val < val:
                    max_val = val
                    max_state = tmp.state.copy()
                alpha = max(alpha, val)
                if beta <= alpha:
                    print(max_val,max_state,"hj")
                    return max_val, max_state
    print(max_val, max_state)
    return max_val, max_state


result=EMM(INITIAL_STATE,0,-np.inf,-np.inf,1)
print(result[0])
GUI_state(result[1])

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

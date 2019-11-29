import numpy as np
from game_state import GameState
from evaluation import evaluate

def p(state):
    if state[GameState.DICE_RESULT] == state[GameState.DICE_RESULT]:
        return 1 / 36
    return 1 / 18


def EMM(state, depth, player_turn):
    if depth == 0 or state.game_over():
        return evaluate(state)
    game = GameState(state)
    if player_turn == 1:
        max_eval = -np.inf
        for position in game.get_possible_states(1):
            eval = p(position) * p(state) * EMM(position, depth - 1, -1)
            max_eval = np.max(max_eval, eval)
        return max_eval
    else:
        min_eval = np.inf
        for position in game.get_possible_states(-1):
            eval = p(position) * p(state) * EMM(position, depth - 1, 1)
            min_eval = np.max(min_eval, eval)
        return min_eval

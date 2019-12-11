import matplotlib.pyplot as plt
import numpy as np
from game_state import GameState
import matplotlib.patches as patches
# white starts at the bottom left
INITIAL_STATE = np.array(
    [0, 2.0, 0, 0, 0, 0, -5.0, 0, -3.0, 0, 0, 0, 5.0, -5.0, 0, 0, 0, 3.0, 0, 5.0, 0, 0, 0, 0, -2.0,
     0, 0, 0, 6, 1])
test_state = np.array(
    [13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, -1.0,
     10, 0, 0, 5, 6])
SOLDIERS_IN_ROW = 12
SOLDIERS_IN_COL = 9
SOLDIER_SIZE = 50
NUM_OF_POSITIONS = 24


def GUI_state(l):
    x = np.arange(0, 1 + 1.0 / SOLDIERS_IN_ROW, 1.0 / SOLDIERS_IN_ROW)
    ax = plt.gca()
    ax.set_facecolor('xkcd:brown')
    for pos in x[1:]:
        plt.plot(np.ones_like(x) * pos - 1.0 / NUM_OF_POSITIONS,
                 (x - 1.0 / 2 / NUM_OF_POSITIONS) * 1.05, 'r')
    # print(l)
    x_rect = 1 + 1 / SOLDIERS_IN_ROW
    y_rect = 0.2
    plt.text(x_rect, y_rect / 2,
             s=str(int(l[GameState.DICE_RESULT])) + "   " + str(int(l[GameState.DICE_RESULT + 1])),
             fontsize=20)
    plt.text(x_rect-0.02, 1-y_rect / 2,
             s="W"+str(int(l[GameState.NUMBER_OF_POSITIONS-1])) + " " + "B"+str(int(abs(l[0]))),
             fontsize=17)
    rect_width = 0.1
    rect_height = 1 - y_rect * 2
    rect = patches.Rectangle((x_rect, y_rect), rect_width, rect_height, linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    eat_pos_x = x_rect + rect_width / 2
    eat_pos_y = 1 - y_rect - 1 / SOLDIERS_IN_ROW - 0.03
    for i in range(GameState.KILLED_SOLDIERS_INDEX, GameState.KILLED_SOLDIERS_INDEX + 2):
        col = "w" * (i == GameState.KILLED_SOLDIERS_INDEX) + "k" * (
                i != GameState.KILLED_SOLDIERS_INDEX)
        for k in range(int(l[i])):
            plt.scatter(eat_pos_x, eat_pos_y, s=SOLDIER_SIZE, c=col)
            eat_pos_y -= 1.0 / SOLDIERS_IN_ROW / 2
    l = l[1:GameState.NUMBER_OF_POSITIONS - 1]
    l_abs = np.abs(l)

    for i in range(len(l)):
        col = "w" * int(l[i] > 0) + "k" * int(l[i] < 0)
        shift = (i >= SOLDIERS_IN_ROW)
        if l[i] == 0:
            continue
        pos_x = x[i * (shift == 0) + (2 * SOLDIERS_IN_ROW - 1 - i) * (shift == 1)]
        for k in range(int(l_abs[i])):
            plt.scatter(pos_x, shift + (-1) ** (shift == 1) * k * 0.5 / SOLDIERS_IN_COL,
                        s=SOLDIER_SIZE,
                        c=col)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.show()


# my_state = GameState(test_state)
# all_states = my_state.get_possible_states(1)
# for state in all_states:
#     GUI_state(state)

# GUI_state(test_state)

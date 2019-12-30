import numpy as np
import cv2
from EMM_manager import GUI, EMM
import Board
from Constants import *
from Board_detection_manager.Soldier_Detection import ImageProcessing

# # Camera settings:
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Debugging uses:
img_test = cv2.imread(
    "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\new_game_manager\\Board_detection_manger\\Soldier_Detection\\test_photos\\WIN_20191212_10_45_09_Pro.jpg",
    cv2.IMREAD_COLOR)


def initialize_board():
    global board
    board = Board.Board()


def get_xy_from_triangle_number(num_triangle, triangle_stack, curr_radius):
    """
    Gets the next spot to put a piece in that triangle
    :param num_triangle: the serial num of triangle
    :param triangle_stack: the stack of pieces in theat triangle
    :param curr_radius: radius of piece to move there
    :return: the x and y position to move to (the middle of the piece)
    """
    num_triangle = num_triangle + 1
    tmp_num_triangle = num_triangle * (num_triangle < 13) \
                       + (25 - num_triangle) * (25 > num_triangle > 12)
    x = A / 2 + A * (tmp_num_triangle - 1) + DELTA * (tmp_num_triangle > 6)
    y_diff = sum([triangle_stack[i][2] for i in range(len(triangle_stack))])*2 + curr_radius
    y = HEIGHT * (num_triangle < 13) + y_diff * (-1) ** (num_triangle < 13)  # from the upper part
    # or from the bottom part
    return [x, y]


def get_eaten_place(stack_board):
    eaten_stack = stack_board[26] + stack_board[27]
    x = WIDTH / 2
    y = HEIGHT / 2
    placed = False
    while not placed:
        placed = True
        for piece in eaten_stack:
            if np.abs((piece[1] - y)) < 2.2 * piece[2]:
                placed = False
                break
        if not placed:
            y = (y + eaten_stack[0][2]) % HEIGHT

    # TODO: make it work with multiple eaten
    return [x, y]


def get_move_from_state(new_state):
    """
    Gets the move done from the old and new states
    :param new_state: the board state after the move
    :return: the move done
    """
    curr_state = board.get_board_position()
    diff = new_state - curr_state  # se difference in pieces locations
    moves = []
    if diff[26] > 0 or diff[27] > 0:  # manage cases we have eaten
        places_eaten = np.where(np.multiply(curr_state[1:25], new_state[1:25]) < 0)
        started_opp = np.where(curr_state[1:25] == -player_turn)
        ended_empty = np.where(new_state[1:25] == 0)
        places_eaten_skipped = np.intersect1d(started_opp, ended_empty)
        for place in places_eaten:
            if len(place) > 0:
                moves.append([place[0], 26 + 1 * (player_turn == 1)])
                diff[place[0] + 1] -= player_turn
        for place in places_eaten_skipped:
            moves.append([place, 26 + 1 * (player_turn == 1)])
            diff[place + 1] -= player_turn
    if diff[26] < 0 or diff[27] < 0:  # manage if we got eaten
        dices = board.get_board_position()[28:30]
        # if dices[0] == dices[1]:  # Double
        #     diff[25 - dices[0]]
        changed = False
        for dice_val in dices:
            diff_index = int((25 - dice_val) * (player_turn == -1) + dice_val * (player_turn == 1))
            if diff[diff_index] == player_turn:
                diff[diff_index] -= player_turn
                moves.append([26 + (player_turn == BLACK), diff_index - 1*(player_turn == BLACK)])
                changed = True
                # print(diff_index, "diff")
        if not changed:
            dice_val = dices[0] + dices[1]
            diff_index = int((25 - dice_val) * (player_turn == -1) + dice_val * (player_turn == 1))
            if diff[diff_index] == -player_turn:
                diff[diff_index] += player_turn
                moves.append([26 + (player_turn == BLACK), diff_index - 1*(player_turn == BLACK)])
                # print(diff_index, "diff")
    while np.sum(np.abs(diff[1:25])) > 0:  # manage regular moves
        moved = np.where(diff*player_turn < 0)[0]
        captured = np.where(diff*player_turn > 0)[0]
        # print (moved, captured)
        moves.append([int(moved[0] - 1), int(captured[0] - 1)])
        diff[moved[0]] += player_turn
        diff[captured[0]] -= player_turn
    return moves


def do_move(all_moves, result_img):  # does the move - showing on GUI
    stack_board = board.get_board_content()
    for action in all_moves:
        piece_to_move = stack_board[action[0] + 1 * (action[0] < 26)].pop()
        spot_to_move_to = action[1]
        color_from = (0, 255, 0)
        color_target = (255, 0, 0)
        if spot_to_move_to == 26 or spot_to_move_to == 27:
            xy_to_move_to = get_eaten_place(stack_board)
            color_from = (0, 0, 255)
            color_target = (0, 0, 255)
        else:
            xy_to_move_to = get_xy_from_triangle_number(spot_to_move_to, stack_board[spot_to_move_to + 1],
                                                        piece_to_move[2])
        stack_board[action[1] + 1 * (action[1] < 26)].append(piece_to_move)
        cv2.circle(result_img, (int(piece_to_move[0]), int(piece_to_move[1])), int(piece_to_move[2]), color_from, 2)
        cv2.circle(result_img, (int(xy_to_move_to[0]), int(xy_to_move_to[1])), int(piece_to_move[2]), color_target,
                   2)


player_turn = 1
initialize_board()
while True:
    dice_roll = input("Dice Result: ")
    dice_result = [int(dice_roll[0]), int(dice_roll[2])]
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    board.update_from_camera(img, dice_result)  # update the Board according to the image
    GUI.GUI_state(board.get_board_position())  # show on GUI
    result = EMM.EMM(board.get_board_position(), 1, player_turn)  # get EMM move
    new_board = result[1]
    GUI.GUI_state(new_board)  # show new board after move
    cv2.waitKey(0)
    move = get_move_from_state(new_board)  # get the move to do
    do_move(move, img)  # make the move (currently only GUI visual)
    cv2.imshow("result", img)  # show move done
    cv2.waitKey(0)
    player_turn = -1 * player_turn


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

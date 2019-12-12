import numpy as np
import cv2
from Board_detection_manger.Soldier_Detection import ImageProcessing
from EMM_manager import GUI,EMM
import matplotlib.pyplot as plt



NUMBER_OF_TRIANGLES = 24
WHITE = 1
BLACK = -1

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

img_test = cv2.imread(
    "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\new_game_manager\\Board_detection_manger\\Soldier_Detection\\test_photos\\WIN_20191212_10_45_09_Pro.jpg",
    cv2.IMREAD_COLOR)

width = 650
height = 620
a = 50
y_middle = height/2
delta = width - NUMBER_OF_TRIANGLES/2 * a

def compare_distance_from_edge(piece):
    return min(np.abs(piece[1] - height), np.abs(piece[1]))


def get_xy_from_triangle_number(num_triangle, triangle_stack, curr_radius):
    # print(num_triangle, triangle_stack, curr_radius)
    num_triangle = num_triangle + 1
    tmp_num_triangle = num_triangle * (num_triangle < 13) \
                       + (25 - num_triangle) * (25 > num_triangle > 12)
    x = a/2 + a*(tmp_num_triangle - 1) + delta * (tmp_num_triangle > 6)
    y_diff = sum([triangle_stack[i][2] for i in range(len(triangle_stack))])*2 + curr_radius
    y = height* (num_triangle < 13) + y_diff * (-1)**(num_triangle < 13) # from the upper part
    # or from the bottom part
    return [x,y]

def locate_pieces(circles, cells, stack_board, COLOR, eaten, eaten_stack):
    for circle in circles[0]:
        x_position = circle[0]
        y_position = circle[1]
        if x_position != 0:
            if (NUMBER_OF_TRIANGLES/4 + 1)*a > x_position > NUMBER_OF_TRIANGLES/4 * a:
                eaten[1 * (COLOR == BLACK)] += 1
                eaten_stack[1 * (COLOR == BLACK)].append(circle)
            else:
                if x_position > NUMBER_OF_TRIANGLES/4 * a:
                    x_position = x_position - delta
                cell = int(x_position/a)
                if y_position > y_middle:
                    cell = int(NUMBER_OF_TRIANGLES - 1 - cell)
                cells[cell] += COLOR
                stack_board[cell + 1].append(circle)



def camera2board(white_circles, black_circles,dice_result):
    stack_board = [[] for i in range(NUMBER_OF_TRIANGLES + 2)]
    eaten_stack = [[] for i in range(2)]
    cells = np.zeros((NUMBER_OF_TRIANGLES))
    eaten = np.zeros((2))
    locate_pieces(white_circles, cells, stack_board, WHITE, eaten, eaten_stack)
    locate_pieces(black_circles, cells, stack_board, BLACK, eaten, eaten_stack)
    cells = list(np.flip(cells))
    cells = [0.0]+cells+[0.0]+list(eaten)+dice_result
    stack_board = stack_board[::-1] + eaten_stack
    global first_run
    if first_run:
        for stack in stack_board:
            stack.sort(key=compare_distance_from_edge)
        first_run = False
    return np.array(cells), stack_board

def get_eaten_place(stack_board):
    eaten_stack = stack_board[26] + stack_board[27]
    x = width / 2
    y = height / 2
    placed = False
    while not placed:
        placed = True
        for piece in eaten_stack:
            if np.abs((piece[1] - y)) < 2.2 * piece[2]:
                placed = False
                break
        if not placed:
            y = (y + eaten_stack[0][2]) % height

    # TODO: make it work withmultiple eaten
    return [x,y]


def get_move_from_state(curr_state, new_state, player_turn):
    diff = new_state - curr_state
    moves = []
    if diff[26]>0 or diff[27]>0:  # we have eaten
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
    if diff[26]<0 or diff[27]<0:
        dices = game_board[28:30]
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
                moves.append([26 + (player_turn == BLACK), diff_index- 1*(player_turn == BLACK)])
                # print(diff_index, "diff")
    while np.sum(np.abs(diff[1:25])) > 0:
        moved=np.where(diff*player_turn<0)[0]
        captured=np.where(diff*player_turn>0)[0]
        # print (moved, captured)
        moves.append([int(moved[0] - 1), int(captured[0]- 1)])
        diff[moved[0]] += player_turn
        diff[captured[0]] -= player_turn
    return moves


global first_run
first_run = True
player_turn = 1
while True:
    dice_roll = raw_input("Dice Result: ")
    dice_result = [int(dice_roll[0]), int(dice_roll[2])]
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    # Display the resulting frame
    # print(img.shape)
    # cv2.imshow('frame', img)
    # cv2.waitKey(0)
    white_circles, black_circles, result_img = ImageProcessing.get_detected_soldiers(img_input=img)
    # white_circles, black_circles, result_img = ImageProcessing.get_detected_soldiers(img_input=img_test)
    game_board, stack_board = camera2board(white_circles, black_circles, dice_result)
    GUI.GUI_state(game_board)
    result = EMM.EMM(game_board, 1, player_turn)
    board = result[1]
    GUI.GUI_state(board)
    cv2.waitKey(0)
    move = get_move_from_state(game_board, board, player_turn)
    for action in move:
        piece_to_move = stack_board[action[0] + 1 * (action[0] < 26)].pop()
        spot_to_move_to = action[1]
        colorFrom = (0, 255, 0)
        colorTarget = (255, 0, 0)
        if spot_to_move_to == 26 or spot_to_move_to == 27:
            xy_to_move_to = get_eaten_place(stack_board)
            colorFrom = (0, 0, 255)
            colorTarget = (0, 0, 255)
        else:
            xy_to_move_to = get_xy_from_triangle_number(spot_to_move_to, stack_board[spot_to_move_to + 1],
                                                    piece_to_move[2])
        stack_board[action[1] + 1 * (action[1] < 26)].append(piece_to_move)
        cv2.circle(result_img, (int(piece_to_move[0]), int(piece_to_move[1])), int(piece_to_move[2]), colorFrom, 2)
        cv2.circle(result_img, (int(xy_to_move_to[0]), int(xy_to_move_to[1])), int(piece_to_move[2]), colorTarget,
                   2)

        # print (xy_to_move_to)
    cv2.imshow("result", result_img)
    cv2.waitKey(0)
    player_turn = -1 * player_turn




# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
from Board_detection_manger.Soldier_Detection import ImageProcessing
from EMM_manager import GUI,EMM



NUMBER_OF_TRIANGLES = 24
WHITE = 1
BLACK = -1


cap = cv2.VideoCapture(0)

img_test = cv2.imread(
    "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\new_game_manager\\Board_detection_manger\\Soldier_Detection\\test_photos\\WIN_20191209_17_00_02_Pro.jpg",
    cv2.IMREAD_COLOR)

width = 650
height = 620
a = 50
y_middle = height/2
delta = width - NUMBER_OF_TRIANGLES/2 * a

def get_xy_from_triangle_number(num_triangle, triangle_stack, curr_radius):
    tmp_num_triangle = num_triangle * (num_triangle < 13) \
                       + (25 - num_triangle) * (25 > num_triangle > 12)
    x = a/2 + a*(tmp_num_triangle - 1) + delta * (tmp_num_triangle > 6)
    y_diff = sum([triangle_stack[i][3] for i in range(len(triangle_stack))])*2 + curr_radius
    y = height* (num_triangle > 12) + y_diff * (-1)**(num_triangle < 12) # from the upper part
    # or from the bottom part
    return [x,y]


def camera2board(white_circles, black_circles,dice_result):
    stack_board = [[] for i in range(NUMBER_OF_TRIANGLES)]
    cells = np.zeros((24))
    for white_circle in white_circles[0]:
        x_position = white_circle[0]
        y_position = white_circle[1]
        if x_position != 0:
            if x_position > NUMBER_OF_TRIANGLES/4 * a:
                x_position = x_position - delta
            cell = int(x_position/a)
            if y_position > y_middle:
                cell = int(NUMBER_OF_TRIANGLES - 1 - cell)
            cells[cell] += WHITE
            stack_board[cell].append(white_circle)

    for black_circle in black_circles[0]:
        x_position = black_circle[0]
        y_position = black_circle[1]
        if x_position != 0:
            if x_position > NUMBER_OF_TRIANGLES / 4 * a:
                x_position = x_position - delta
            cell = int(x_position / a)
            if y_position > y_middle:
                cell = int(NUMBER_OF_TRIANGLES - 1 - cell)
            cells[cell] += BLACK
            stack_board[cell].append(black_circle)
    cells = np.flip(cells)
    cells = list(cells)
    # TODO: add real data
    # eaten=get_eaten()
    eaten=[0.0, 0.0]
    cells=[0.0]+cells+[0.0]+eaten+dice_result
    return np.array(cells), stack_board

def get_move_from_state(curr_state, new_state, player_turn):
    diff = new_state - curr_state
    moves = []
    while np.sum(np.abs(diff)) > 0:
        moved = np.where(diff <= -player_turn)
        captured = np.where(diff >= player_turn)
        moves.append([int(moved[0]), int(captured[0])])
        diff[moved[0]] += player_turn
        diff[captured[0]] -= player_turn
    return moves


while True:
    player_turn = 1
    dice_roll = raw_input("Dice Result: ")
    dice_result = [int(dice_roll[0]), int(dice_roll[2])]
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    # Display the resulting frame
    # cv2.imshow('frame', img)
    # cv2.waitKey(0)

    white_circles, black_circles, result_img = ImageProcessing.get_detected_soldiers(img_input=img_test)
    # cv2.imshow("dda", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    game_board, stack_board = camera2board(white_circles, black_circles,dice_result)
    GUI.GUI_state(game_board)
    result = EMM.EMM(game_board, 1, 1)
    board = result[1]
    GUI.GUI_state(board)
    cv2.waitKey(0)
    move = get_move_from_state(game_board, board, player_turn)
    print ("HELOOOO")
    for action in move:
        print(action)
        piece_to_move = stack_board[action[0]].pop()
        spot_to_move_to = action[1]
        print(piece_to_move)
        xy_to_move_to = get_xy_from_triangle_number(spot_to_move_to, stack_board[spot_to_move_to],
                                                    piece_to_move[2])
        stack_board[action[1]].append(piece_to_move)
        cv2.circle(result_img, (int(piece_to_move[0]), int(piece_to_move[1])), int(piece_to_move[2]), (0, 255, 0), 2)
        cv2.circle(result_img, (int(xy_to_move_to[0]), int(xy_to_move_to[1])), int(piece_to_move[2]), (255, 0, 0),
                   2)
    cv2.imshow("result", result_img)
    cv2.waitKey(0)
    break




# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

from new_game_manager.Constants import *
from Board_detection_manager.Soldier_Detection import ImageProcessing


def get_empty_xy_from_triangle_number(num_triangle, num_in_line):
    """
    Gets the next spot to put a piece in that triangle
    :param num_triangle: the serial num of triangle
    :return: the x and y position to move to (the middle of the piece)
    """
    tmp_num_triangle = num_triangle * (num_triangle < 13) \
                       + (25 - num_triangle) * (25 > num_triangle > 12)
    x = A / 2 + A * (tmp_num_triangle - 1) + DELTA * (tmp_num_triangle > 6)
    y_diff = APPROX_RADIUS * (num_in_line*2 + 1)
    y = HEIGHT * (num_triangle < 13) + y_diff * (-1) ** (num_triangle < 13)  # from the upper part
    # or from the bottom part
    return [x, y]


def move_motor_temp(start, end):
    print(start[0], start[1])
    print("to")
    print(end[0], end[1])


def get_required_positions():
    white_positions = [1, 1, 12, 12, 12, 12, 12, 17, 17, 17, 19, 19, 19, 19, 19]
    black_positions = [6, 6, 6, 6, 6, 8, 8, 8, 13, 13, 13, 13, 13, 24, 24]
    white_xy_positions = []
    black_xy_positions = []
    count = 0
    pre_pos = 0
    for pos in white_positions:
        if pos == pre_pos:
            count += 1
        else:
            count = 0
        white_xy_positions.append(get_empty_xy_from_triangle_number(pos, count))

    for pos in black_positions:
        if pos == pre_pos:
            count += 1
        else:
            count = 0
        black_xy_positions.append(get_empty_xy_from_triangle_number(pos, count))

    return white_xy_positions, black_xy_positions


def is_coliding(circle, pos):
    x_pos = circle[0]
    y_pos = circle[1]
    radius = circle[2]
    return (x_pos - pos[0])**2 + (y_pos - pos[2])**2 <= radius**2


def find_closest(circles, pos):
    closest_dist = 99999
    closest_circle = circles[0]
    for circle in circles:
        x_pos = circle[0]
        y_pos = circle[1]
        if (x_pos - pos[0]) ** 2 + (y_pos - pos[2]) ** 2 <= closest_dist:
            closest_dist = (x_pos - pos[0]) ** 2 + (y_pos - pos[2]) ** 2
            closest_circle = circle
    return closest_circle


def fix_board():
    white_capt = [0]*15
    black_capt = [0]*15
    fixed = False
    while not fixed:
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        white_circles, black_circles, result_img = ImageProcessing.get_detected_soldiers(img_input=img)
        white_positions, black_positions = get_required_positions()
        for i in range(len(white_capt)):
            if white_capt[i] == 1:
                continue
            white_pos = white_positions[i]
            for circle in black_circles[0]:
                if is_coliding(circle, white_pos):
                    motor_start = [circle[0], circle[1]]
                    for j in range(24):
                        if black_capt[len(black_capt) - j - 1] == 0:
                            black_capt[len(black_capt) - j - 1] = 1
                            motor_end = black_positions[len(black_positions) - j]
                    move_motor_temp(motor_start, motor_end)
                    ##MOVE FROM START TO END##
            for circle in white_circles:
                if is_coliding(circle, white_pos):
                    motor_start = [circle[0], circle[1]]
                    motor_end = white_pos
                    break
            if motor_end is not white_pos:
                closest_circle = find_closest(white_circles, white_pos)
                motor_start = [closest_circle[0], closest_circle[1]]
                motor_end = white_pos
                move_motor_temp(motor_start, motor_end)
                ##MOVE FROM START TO END##
                white_circles.remove(closest_circle)
        # All whites in place
        for i in range(len(black_capt)):
            if black_capt[i] == 1:
                continue
            black_pos = black_positions[i]
            for circle in black_circles:
                if is_coliding(circle, black_pos):
                    motor_start = [circle[0], circle[1]]
                    motor_end = black_pos
                    break
            if motor_end is not black_pos:
                closest_circle = find_closest(black_circles, black_pos)
                motor_start = [closest_circle[0], closest_circle[1]]
                motor_end = black_pos
                move_motor_temp(motor_start, motor_end)
                ##MOVE FROM START TO END##
                black_circles.remove(closest_circle)

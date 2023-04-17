'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import time
import copy

def count_black_white(board):
    b = 0
    w = 0
    for row in board:
        for cell in row:
            if cell == 'b':
                b += 1
            elif cell == 'w':
                w += 1
    return b, w

def get_possible_moves(board, color):
    possible_moves = []
    center = (9, 9)
    b, w = count_black_white(board)

    # Check if this is the first move
    if color == 'w' and board == [[0 for _ in range(19)] for _ in range(19)]:
        # First move must be the center
        possible_moves.append(center)
        return possible_moves



    # Check if it's the second move of white
    if color == 'w' and board[center[0]][center[1]] == 'w' and b == 1 and w == 1:
        # Second move must be at least 3 intersections away from center
        for row in range(6, 13):
            for col in range(6, 13):
                if row >= 7 and row <= 11 and col >= 7 and col <= 11:
                    continue
                if board[row][col] == 0:
                    possible_moves.append((row, col))

    else:
        for row in range(19):
            for col in range(19):
                if board[row][col] == 0:
                    possible_moves.append((row, col))

    # print(possible_moves)
    return possible_moves

def update_board(board, move, color, white_captured, black_captured):
    i, j = move
    new_board = copy.deepcopy(board)
    new_board[i][j] = color

    # Check for captures in horizontal direction
    if j <= 15 and new_board[i][j:j+4] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        new_board[i][j+1: j+3] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    if j >= 3 and new_board[i][j-3: j+1] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        new_board[i][j-2:j] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    # Check for captures in vertical direction
    if i <= 15 and [new_board[i][j], new_board[i+1][j], new_board[i+2][j], new_board[i+3][j]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i+1][j], new_board[i+2][j]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    if i >= 3 and [new_board[i-3][j], new_board[i-2][j], new_board[i-1][j], new_board[i][j]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i-2][j], new_board[i-1][j]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    # Check for captures in diagonal direction (top-left to bottom-right)
    if i <= 15 and j <= 15 and [new_board[i][j], new_board[i+1][j+1], new_board[i+2][j+2], new_board[i+3][j+3]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i+1][j+1], new_board[i+2][j+2]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    if i >= 3 and j >= 3 and [new_board[i-3][j-3], new_board[i-2][j-2], new_board[i-1][j-1], new_board[i][j]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i-2][j-2], new_board[i-1][j-1]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    # Check for captures in diagonal direction (top-right to bottom-left)
    if i <= 15 and j >= 3 and [new_board[i+3][j-3], new_board[i+2][j-2], new_board[i+1][j-1], new_board[i][j]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i+2][j-2], new_board[i+1][j-1]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    if i >= 3 and j <= 15 and [new_board[i][j], new_board[i-1][j+1], new_board[i-2][j+2], new_board[i-3][j+3]] == [color, get_opponent_color(color), get_opponent_color(color), color]:
        [new_board[i-1][j+1], new_board[i-2][j+2]] = [0] * 2
        if color == 'b':
            black_captured += 2
        else:
            white_captured += 2

    return new_board, white_captured, black_captured

def is_valid_cell(row, col):
    return row >= 0 and row < 19 and col >= 0 and col < 19

def get_num_open_lines(board, color, length):

    num_open_lines = 0

    # Check horizontal lines
    for row in range(19):
        for col in range(15):
            line = board[row][col:col+5]
            if line.count(color) == 5 - length and line.count(0) == length:
                num_open_lines += 1

    # Check vertical lines
    for col in range(19):
        for row in range(15):
            line = [board[row+i][col] for i in range(5)]
            if line.count(color) == 5 - length and line.count(0) == length:
                num_open_lines += 1

    # Check diagonal lines (top-left to bottom-right)
    for row in range(15):
        for col in range(15):
            line = [board[row+i][col+i] for i in range(5)]
            # print(line)
            # print(line)
            if line.count(color) == 5 - length and line.count(0) == length:
                num_open_lines += 1

    # Check diagonal lines (bottom-left to top-right)
    for row in range(4, 19):
        for col in range(15):
            line = [board[row-i][col+i] for i in range(5)]
            if line.count(color) == 5 - length  and line.count(0) == length:
                num_open_lines += 1

    return num_open_lines

def has_won(board, color):
    # Check horizontal lines
    for row in range(19):
        for col in range(15):
            if board[row][col] == color and board[row][col+1] == color and \
               board[row][col+2] == color and board[row][col+3] == color and \
               board[row][col+4] == color:
                return True

    # Check vertical lines
    for row in range(15):
        for col in range(19):
            if board[row][col] == color and board[row+1][col] == color and \
               board[row+2][col] == color and board[row+3][col] == color and \
               board[row+4][col] == color:
                return True

    # Check diagonal lines (top-left to bottom-right)
    for row in range(15):
        for col in range(15):
            if board[row][col] == color and board[row+1][col+1] == color and \
               board[row+2][col+2] == color and board[row+3][col+3] == color and \
               board[row+4][col+4] == color:
                return True

    # Check diagonal lines (bottom-left to top-right)
    for row in range(4, 19):
        for col in range(15):
            if board[row][col] == color and board[row-1][col+1] == color and \
               board[row-2][col+2] == color and board[row-3][col+3] == color and \
               board[row-4][col+4] == color:
                return True

    return False

def vulnerability_to_capture(board, color, opponent_color, I):
    vulnerable_penalty = 0
    for i in range(19):
        for j in range(19):
            if board[i][j] == color:
                # print(i, j)
                # Check for horizontal lines of 3 or more stones with an adjacent opponent stone
                #  ..bww...
                if j >= 1 and j <= 16 and board[i][j - 1] == opponent_color and board[i][j: j+3] == [color] * 2 + [0]:
                    vulnerable_penalty += I
                #  ..wwb...
                if j >= 2 and j <= 17 and board[i][j + 1] == opponent_color and board[i][j - 2:j + 1] == [0] + [color] * 2:
                    vulnerable_penalty += I

                # Check for vertical lines of 3 or more stones with an adjacent opponent stone
                #  ..b...
                #  ..w...
                #  ..w...
                if i >= 1 and i<= 16 and board[i - 1][j] == opponent_color and [board[i][j], board[i+1][j], board[i+2][j]]  == [color] * 2 + [0]:
                    vulnerable_penalty += I
                #  ..w...
                #  ..w...
                #  ..b...
                if i >= 2 and i <= 17 and board[i + 1][j] == opponent_color and [board[i-2][j], board[i-1][j], board[i][j]] == [0] + [color] * 2:
                    vulnerable_penalty += I

                # Check for diagonal lines (top-left to bottom-right) of 3 or more stones with an adjacent opponent stone
                #  b.....
                #  .w....
                #  ..w...
                if i >= 1 and j >= 1 and i <= 16 and j <= 16 and board[i - 1][j - 1] == opponent_color and [board[i][j], board[i + 1][j + 1], board[i + 2][j + 2]] == [color] * 2 + [0]:
                    vulnerable_penalty += I
                #  ...w..
                #  ....w.
                #  .....b
                if i >= 2 and j >= 2 and i <= 17 and j <= 17 and board[i + 1][j + 1] == opponent_color and [board[i - 2][j - 2], board[i - 1][j - 1], board[i][j]] == [0] + [color] * 2:
                    vulnerable_penalty += I

                # Check for diagonal lines (top-right to bottom-left) of 3 or more stones with an adjacent opponent stone
                #  .....b
                #  ....w.
                #  ...w..
                if i >= 1 and j <= 17 and i <= 16 and j >= 2 and board[i - 1][j + 1] == opponent_color and [board[i][j], board[i + 1][j - 1], board[i + 2][j - 2]] == [color] * 2 + [0]:
                    vulnerable_penalty += I
                #  ..w...
                #  .w....
                #  b.....
                if i <= 17 and j >= 1 and i>=3 and j<= 16 and board[i + 1][j - 1] == opponent_color and [board[i][j], board[i - 1][j + 1], board[i - 2][j + 2]] == [color] * 2 + [0]:
                    vulnerable_penalty += I

    return vulnerable_penalty

def open_3(board, opponent_color, III):
    vulnerable_penalty = 0
    for i in range(0,19):
        for j in range(0,19):
            if board[i][j] == opponent_color:
            # Check for horizontal lines
                # .bbb.
                if j >= 1 and j <= 15 and [board[i][j - 1], board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3]] == [0] + [opponent_color] * 3 + [0]:
                    vulnerable_penalty += III
            # Check for vertical lines
                # .....
                # .b...
                # .b...
                # .b...
                # .....
                if i >= 1 and i <= 15 and [board[i - 1][j], board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j]] == [0] + [opponent_color] * 3 + [0]:
                    vulnerable_penalty += III

            # Check for diagonal lines (top-left to bottom-right)
                # ......
                # .b....
                # ..b...
                # ...b..
                # ......
                if i >= 1 and j >= 1 and i <= 15 and j <= 15 and [board[i - 1][j - 1], board[i][j], board[i + 1][j + 1], board[i + 2][j + 2], board[i + 3][j + 3]] == [0] + [opponent_color] * 3 + [0]:
                    vulnerable_penalty += III

            # Check for diagonal lines (top-right to bottom-left)
                # .....
                # ...b.
                # ..b..
                # .b...
                # .....
                if i >= 1 and j <= 17 and i <= 15 and j >= 3 and [board[i - 1][j + 1], board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3]] == [0] + [opponent_color] * 3 + [0]:
                    vulnerable_penalty += III

    return vulnerable_penalty

def vulnerability_of_open_4(board, opponent_color, II):
    vulnerable_penalty = 0
    for i in range(0,19):
        for j in range(0,19):
            if board[i][j] == opponent_color:
            # Check for horizontal lines
                # .b.bb...
                if j >= 1 and j <= 14 and [board[i][j - 1], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]] == [0, 0] + [opponent_color] * 2 + [0]:
                    vulnerable_penalty += II
                # .bb.b...
                if j >= 1 and j <= 14 and [board[i][j - 1], board[i][j + 1], board[i][j + 2], board[i][j + 3], board[i][j + 4]] == [0] + [opponent_color] + [0] + [opponent_color] + [0]:
                    vulnerable_penalty += II
            # Check for vertical lines
                # .....
                # .b...
                # .....
                # .b...
                # .b...
                # .....
                if i >= 1 and i <= 14 and [board[i - 1][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]] == [0, 0] + [opponent_color] * 2 + [0]:
                    vulnerable_penalty += II
                # .....
                # .b...
                # .b...
                # .....
                # .b...
                # .....
                if i >= 1 and i <= 14 and [board[i - 1][j], board[i + 1][j], board[i + 2][j], board[i + 3][j], board[i + 4][j]] == [0] + [opponent_color] + [0] + [opponent_color] + [0]:
                    vulnerable_penalty += II
            # Check for diagonal lines (top-left to bottom-right)
                # ......
                # .b....
                # ......
                # ...b..
                # ....b.
                # ......
                if i >= 1 and j >= 1 and i <= 14 and j <= 14 and [board[i - 1][j - 1], board[i + 1][j + 1], board[i + 2][j + 2], board[i + 3][j + 3], board[i + 4][j + 4]] == [0, 0] + [opponent_color] * 2 + [0]:
                    vulnerable_penalty += II
                # ......
                # .b....
                # ..b...
                # ......
                # ....b.
                # ......
                if i >= 1 and j >= 1 and i <= 14 and j <= 14 and [board[i - 1][j - 1], board[i + 1][j + 1], board[i + 2][j + 2], board[i + 3][j + 3], board[i + 4][j + 4]] == [0] + [opponent_color] + [0] + [opponent_color] + [0]:
                    vulnerable_penalty += II
            # Check for diagonal lines (top-right to bottom-left)
                # ......
                # ....b.
                # ......
                # ..b...
                # .b....
                # ......
                if i >= 1 and j <= 17 and i <= 14 and j >= 4 and [board[i - 1][j + 1], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3], board[i + 4][j - 4]] ==  [0, 0] + [opponent_color] * 2 + [0]:
                    vulnerable_penalty += II
                # ......
                # ....b.
                # ...b..
                # ......
                # .b....
                # ......
                if i >= 1 and j <= 17 and i <= 14 and j >= 4 and [board[i - 1][j + 1], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3], board[i + 4][j - 4]] == [0] + [opponent_color] + [0] + [opponent_color] + [0]:
                    vulnerable_penalty += II

    return vulnerable_penalty


def evaluate_board(new_board, my_color, i_got_captured, oppo_got_captured, my_captured_this_round):
    # Initialize the weights for each factor

    A = 1000 if oppo_got_captured <= 6 else 110000 # my_captured_this_round
    B = 1500  # one_to_win
    C = 100000  # one_to_lose
    D = 400   # two_to_win
    E = 600   # two_to_lose
    F = 200   # three_to_win
    G = 300   # three_to_lose
    H = 100   # four_to_win
    I = 1500 if i_got_captured <= 6 else 15000 # Penalty for being vulnerable to capture
    II = 6000 # one_to_open_four
    III = 5000 # open_three
    IIII = 1000

    # Determine the opponent's color
    opponent_color = get_opponent_color(my_color)

    # Check if the program has won the game
    if has_won(new_board, my_color):
        return 100000000000

    # Check if the opponent has won the game
    if has_won(new_board, opponent_color):
        return -100000000000

    # Initialize the values for each factor
    # his_captured = black_captured if color == 'w' else white_captured
    # my_captured = white_captured if color == 'b' else black_captured
    one_to_win = get_num_open_lines(new_board, my_color, 1)
    one_to_lose = get_num_open_lines(new_board, opponent_color, 1)
    two_to_win = get_num_open_lines(new_board, my_color, 2)
    two_to_lose = get_num_open_lines(new_board, opponent_color, 2)
    three_to_win = get_num_open_lines(new_board, my_color, 3)
    three_to_lose = get_num_open_lines(new_board, opponent_color, 3)
    four_to_win = get_num_open_lines(new_board, my_color, 4)
    # Calculate the penalty for vulnerability to capture
    capture_penalty = vulnerability_to_capture(new_board, my_color, opponent_color, I)
    one_to_open_four = vulnerability_of_open_4(new_board, opponent_color, II)
    my_open_three = open_3(new_board, my_color, IIII)
    oppo_open_three = open_3(new_board, opponent_color, III)

    # test = get_num_open_lines(board, opponent_color, 5)


    # # Calculate the value of the heuristic
    # value = A*(his_captured - my_captured) + B*(one_to_win) - C*(one_to_lose) \
    #       + D*(two_to_win) - E*(two_to_lose) + F*(three_to_win) - G*(three_to_lose) \
    #       + H*(four_to_win)
    #
    # return value

    # Calculate the value of the heuristic
    value = (A* my_captured_this_round) + (B * one_to_win) - (C * one_to_lose) \
          + (D * two_to_win) - (E * two_to_lose) + (F * three_to_win) - (G * three_to_lose) \
          + (H * four_to_win) - capture_penalty - one_to_open_four + my_open_three - oppo_open_three

    # print(one_to_lose)
    # return value
    # return value, one_to_win, one_to_lose, two_to_win, two_to_lose, three_to_win, three_to_lose, four_to_win, capture_penalty, one_to_open_four, open_three
    return value


def get_opponent_color(color):
    if color == 'w':
        return 'b'
    elif color == 'b':
        return 'w'

def alpha_beta(board, my_color, search_depth, alpha, beta, time_left, white_captured, black_captured, my_captured_this_round, i_got_captured, oppo_got_captured, maximizing_player):
    # check if we have reached the search depth or if there is no time left
    if search_depth == 0 or time_left <= 5:
        # print('return')
        return evaluate_board(board, my_color, i_got_captured, oppo_got_captured, my_captured_this_round)

    # loop through all possible moves
    for move in get_possible_moves(board, my_color):
        # print(move)
        # make a copy of the board and update it with the move
        new_board, new_white_captured, new_black_captured = update_board(board, move, my_color, white_captured, black_captured)
        my_captured_this_round = new_white_captured - white_captured if color == 'b' else new_black_captured - black_captured

        (i_got_captured, oppo_got_captured) = (new_white_captured, new_black_captured) if my_color == 'w' else (new_black_captured, new_white_captured)
        # print(new_board)
        # calculate the score for this move using the alpha-beta algorithm
        if maximizing_player:
            # print('M')
            score = alpha_beta(new_board, color, search_depth - 1, alpha, beta, time_left, white_captured, black_captured, my_captured_this_round, i_got_captured, oppo_got_captured, maximizing_player=False)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        else:
            # print('m')
            score = alpha_beta(new_board, get_opponent_color(color), search_depth - 1, alpha, beta, time_left, black_captured, white_captured, my_captured_this_round, i_got_captured, oppo_got_captured, maximizing_player=True)
            if score < beta:
                beta = score
            if beta <= alpha:
                break

    # return the best score depending on the player
    if maximizing_player:
        return alpha
    else:
        return beta

def determine_depth(time_left):
    if time_left >= 70:
        search_depth = 4
    elif time_left >= 10:
        search_depth = 2
    else:
        search_depth = 1

    return search_depth

def get_next_move(board, my_color, time_left, white_captured, black_captured):
    # define search depth, initial alpha and beta values
    search_depth = determine_depth(time_left)
    # search_depth = 2
    alpha = float('-inf')
    beta = float('inf')
    # initialize the best move and score
    best_move = None
    best_score = None
    # loop through all possible moves
    for move in get_possible_moves(board, my_color):
        # make a copy of the board and update it with the move
        new_board, new_white_captured, new_black_captured = update_board(board, move, my_color, white_captured, black_captured)
        my_captured_this_round = new_white_captured - white_captured if color == 'w' else new_black_captured - black_captured
        (i_got_captured, oppo_got_captured) = (new_white_captured, new_black_captured) if my_color == 'b' else (new_black_captured, new_white_captured)

        # calculate the score for this move using the alpha-beta algorithm
        # score = alpha_beta(new_board, my_color, 1, alpha, beta, time_left, white_captured, black_captured, my_captured_this_round, i_got_captured, oppo_got_captured, maximizing_player=True)
        # score, vulnerable_penalty = evaluate_board(new_board, my_color, new_white_captured, new_black_captured)

        # score, one_to_lose = evaluate_board(new_board, my_color, i_got_captured, oppo_got_captured, my_captured_this_round)

        score = evaluate_board(new_board, my_color, i_got_captured, oppo_got_captured, my_captured_this_round)

        # if move == (6, 9):
        #     print(move)
        #     print('one_to_win: ', one_to_win)
        #     print('one_to_lose: ', one_to_lose)
        #     print('two_to_win: ', two_to_win)
        #     print('two_to_lose: ', two_to_lose)
        #     print('three_to_win: ', three_to_win)
        #     print('three_to_lose: ', three_to_lose)
        #     print('four_to_win: ', four_to_win)
        #     print('capture_penalty: ', capture_penalty)
        #     print('one_to_open_four: ', one_to_open_four)
        #     print('open_three: ', open_three)
        #     # print('new_board: ', new_board)
        #     print('score: ', score)
        # if move == (1, 5):
            # print(move)
        #     print('one_to_win: ', one_to_win)
        #     print('one_to_lose: ', one_to_lose)
        #     print('two_to_win: ', two_to_win)
        #     print('two_to_lose: ', two_to_lose)
        #     print('three_to_win: ', three_to_win)
        #     print('three_to_lose: ', three_to_lose)
        #     print('four_to_win: ', four_to_win)
        #     print('capture_penalty: ', capture_penalty)
        #     print('one_to_open_four: ', one_to_open_four)
        #     print('open_three: ', open_three)
        #     # print('new_board: ', new_board)
        #     print('score: ', score)

        # update the best move and score if this move has a higher score
        if best_score is None or score > best_score:
            best_move = move
            best_score = score

    # return the best move in the required format
    # return convert_move_to_output_format(best_move)
    return best_move



# Initialize the board with empty cells
BOARD_SIZE = 19
board = [[0 for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]

row_index = {i: 19-i for  i in range(19)}
column_index = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',
                7:'H', 8:'J', 9:'K', 10:'L', 11:'M', 12:'N', 13:'O',
                14:'P', 15:'Q', 16:'R', 17:'S', 18:'T'}


# Read the input from the input.txt file
with open('input5.txt', 'r') as input_file:
    color = input_file.readline().strip().upper()
    my_color = 'w' if color == 'WHITE' else 'b'
    time_left = float(input_file.readline().strip())
    captured = input_file.readline().strip().split(',')
    white_captured = int(captured[0])
    black_captured = int(captured[1])

    for i in range(BOARD_SIZE):
        row = input_file.readline().strip()
        for j in range(BOARD_SIZE):
            if row[j] == 'w':
                board[i][j] = 'w'
            elif row[j] == 'b':
                board[i][j] = 'b'

move = get_next_move(board, my_color, time_left, white_captured, black_captured)
# Check if this is the first move of white
# if time_left == 100 and my_color == 'w':
#     if board == [[0 for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]:
#         move = (9, 9)
#     else:
#         move = get_next_move(board, my_color, time_left, white_captured, black_captured)
# else:
#     move = get_next_move(board, my_color, time_left, white_captured, black_captured)
print('move: ', move)
print(str(row_index[move[0]]) + str(column_index[move[1]]))


# Write output file
with open('output.txt', 'w') as f:
    f.write(str(row_index[move[0]]) + str(column_index[move[1]]))
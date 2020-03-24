from os.path import join

from numpy import random
import numpy as np


def is_same_row(mat, char):
    winned = False
    for line in mat:
        winned = winned or list(line).count(char) == 3
    return winned


def is_same_coulnm(mat, char):
    winned = False
    transpose = np.array(mat).transpose()
    for coulnm in transpose:
        winned = winned or list(coulnm).count(char) == 3
    return winned


def is_same_diag(mat, char):
    winned = False
    flip = np.fliplr(mat)
    winned = winned or list(np.array(mat).diagonal()).count(char) == 3 or list(flip.diagonal()).count(char) == 3
    return winned


def is_finished(mat, char):
    return is_same_row(mat, char) or is_same_coulnm(mat, char) or is_same_diag(mat, char)


def print_game(mat):
    mat = np.pad(mat, pad_width=1, mode='constant', constant_values='#')
    print("game status:")
    for line in mat:
        print(*line)


def you_win(mat):
    return is_finished(mat, "X")


def computer_win(mat):
    return is_finished(mat, "O")


def change_board(mat, num, char):
    num = int(num)
    num = num - 1
    row = num // 3
    column = num % 3
    mat[row][column] = char


def check_board(mat, num):
    num = int(num)
    num = num - 1
    row = num // 3
    column = num % 3
    if num >= 0 and num <= 8:
        return mat[row][column] == " "
    else:
        return False


def can_lose(mat):
    for i in range(1,10):
        b = np.array(mat)
        if check_board(b, i):
            change_board(b, i, 'X')
            if is_finished(b, 'X'):
                return True
    return False


def rand_act(_actions):
    a_list = []
    for i in range(0, len(_actions)):
        if _actions[i] > -100:
            a_list.append(i)
    if len(a_list) == 0:
        return -1
    act = np.random.randint(0, len(a_list))
    return a_list[act]


def print_Q(mat):
    s = game_to_state(mat)
    print_game(mat)
    print("Q value for curr board:")
    print(*Q[s])
    print("R value for curr board:")
    print(*R[s])


# using the Q learning algo
def computer_play(mat):
    state = game_to_state(mat)

    # learning rate
    alpha = 0.3
    # gamme
    gamma = 0.5
    # learning rate
    LR = 0.5

    def exploit():
        actions = Q[state]
        best_action = np.argmax(actions)
        while not check_board(mat, best_action + 1):
            Q[state][best_action] = -150
            actions = Q[state]
            best_action = np.argmax(actions)
        return best_action

    def explore():
        actions = Q[state]
        best_action = rand_act(actions)
        if best_action == -1:
            return exploit()
        else:
            while not check_board(mat, best_action + 1):
                Q[state][best_action] = -150
                best_action = explore()
            return best_action

    if random.uniform(0, 1) < alpha:
        # Explore: select a random action
        best_action = explore()
    else:
        # Exploit: select the action with max value (future reward)
        best_action = exploit()
    change_board(mat, best_action + 1, 'O')

    # update the Q table
    new_board = new_board_after_state(mat)
    new_state = game_to_state(new_board)

    # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
    Q[state][best_action] = (1-LR) * Q[state][best_action] + LR * (R[state][best_action] + gamma * np.max(Q[new_state]) - Q[state][best_action])


def rival_act(board):
    act = -1
    for i in range(1,10):
        b = np.array(board)
        if check_board(b, i):
            change_board(b, i, 'X')
            if is_finished(b, 'X'):
                act = i - 1
                break
    if act == -1:
        state = game_to_state(board)
        actions = R[state]
        act = rand_act(actions)
    return act


def rival_action_play(board):
    for i in range(1,10):
        b = np.array(board)
        if check_board(b, i):
            change_board(b, i, 'X')
            if is_finished(b, 'X'):
                return i
    act = np.random.randint(1, 10)
    while not check_board(board, act):
        act += 1
        act = act % 10
    return act


def new_board_after_state(mat):
    board = np.array(mat)
    rival_action = rival_act(board)
    change_board(board, rival_action + 1, 'X')
    return board


game_rules = np.array([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']])


def game_to_state(board):
    number = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == "X":
                number = number + 2 * (3 ** (3 * i + j))
            elif board[i][j] == "O":
                number = number + 1 * (3 ** (3 * i + j))
    return number


def state_to_game(number):
    board = np.array([[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
    List = list(np.base_repr(number, base=3))
    List.reverse()
    List = List + (['0']*(9 - len(List)))
    i = 0
    j = 0
    for char in List:

        if char == "2":
            board[i][j] = "X"
        elif char == "1":
            board[i][j] = "O"
        j += 1
        if j > 0 and j % 3 == 0:
            i += 1
            j = 0
    return board


n = game_to_state([['X', ' ', 'X'], [' ', ' ', 'O'], [' ', '', ' ']])
b = state_to_game(n)

# R matrix
R = np.zeros((3 ** 9, 9))

# set up the rewards
index = 0
for R_line in R:
    if index % 1000 == 0:
        print("we are at the ", int(index/1000), "/ 19 index of creating the R matrix (env)")
    for action in range(0,9):
        board = state_to_game(index)
        if is_finished(board,'X'):
            R_line[action] = -100
            continue
        elif is_finished(board, 'O'):
            R_line[action] = 100
            continue
        else:
            if not check_board(board,action + 1):
                R_line[action] = -150
                continue
            else:
                change_board(board, action + 1, 'O')
                if is_finished(board, 'O'):
                    R_line[action] = 100
                elif can_lose(board):
                    R_line[action] = -100
    index += 1

# Q matrix
Q = np.zeros((3 ** 9, 9))



def play_game():
    game = np.array([[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
    step = 0
    print_game(game_rules)
    print('set up is ready, play game')
    print('you are ''X''')
    pre_game = np.array(game)
    while (not (you_win(game) or computer_win(game))) and step < 9:
        if step % 2 == 0:
            num = input("Enter the location to play: ")
            if num == "Q":
                print_Q(pre_game)
                step += -1
            else:
                while not num.isdigit() or not check_board(game, num):
                    num = input("location is illegal or not a number, play again: ")
                num = int(num)
                change_board(game, num, 'X')
                print_game(game)
        else:
            pre_game = np.array(game)
            print("computer turn")
            computer_play(game)
            print_game(game)
        step += 1

    if you_win(game):
        print("you win!")
    elif computer_win(game):
        print("computer win!")
    else:
        print("it's a draw")


def play_random_game():
    game = np.array([[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']])
    step = 0
    while (not (you_win(game) or computer_win(game))) and step < 9:
        if step % 2 == 0:
            # good play
            num = rival_action_play(game)
            change_board(game, num, 'X')
            # random play
                # num = np.random.randint(1, 9)
                # while not check_board(game, num):
                #   num += 1
                #   num = num % 10
        else:
            computer_play(game)
        step += 1


def play_random():
    for i in range(1, 2001):
        if i % 150 == 0:
            print("I played with myself", i, "games")
        play_random_game()
    print("I am ready to play!")


print("hey, welcome to the best AI tic-tac-toe game")
print("let's learn me how to play")
play = True
while play:
    keep_play = input("keep play ? [Y/N], you can also enter L for me to learn on myself: ")
    while keep_play != "Y" and keep_play != "N" and keep_play != "L":
        keep_play = input("please enter [Y/N]: ")
    if keep_play == "Y":
        play_game()
    elif keep_play == "N":
        play = False
    else:
        play_random()

print("bye bye")

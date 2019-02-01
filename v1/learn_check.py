from ptable import PredictionTable, learning
from smart_tictactoe import *

# PLAYER X FIRST

p_table = PredictionTable(learning_rate=0.1)
print("TOTAL", p_table.step, "ROUND PLAYED")
board = AutoTicTacToe()
smart_o = SmartO(p_table)
smart_x = SmartX(p_table)
smart_ox = SmartOX(p_table)

MAX=1000000
STEP=10000
# MAX=1000
# STEP=100
for step in range(0, MAX, STEP):
    for step1 in range(STEP):
        (result_board, winner) = board.play_game()
        p_table.next_step()
        learning(p_table, result_board, winner)

    p_table.save('p_table.dat')

    count = {'O': 0, 'X': 0, '=': 0}
    for step1 in range(1000):
        (result_board, winner) = smart_o.play_game()
        count[winner] += 1
    print("STEP", step, "SMART_O", count)

    count = {'O': 0, 'X': 0, '=': 0}
    for step1 in range(1000):
        (result_board, winner) = smart_x.play_game()
        count[winner] += 1
    print("STEP", step, "SMART_X", count)

    (result_board, winner) = smart_ox.play_game()
    print("STEP", step, "SMART_OX", result_board, "WINNER", winner)
    if winner != '=':
        smart_ox.print_board(result_board)



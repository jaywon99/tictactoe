import sys
import random

from ptable import PredictionTable 
from smart_tictactoe import *

# TIC TAC TOE BOARD
# 1 | 2 | 3
# ---------
# 4 | 5 | 6
# ---------
# 7 | 8 | 9
#
# 여기서 여짓것 play한 위치 (1~9)를 순서로 넣는다.

p_table = PredictionTable(learning_rate=0.5)
p_table.load('p_table.dat')
print("TOTAL", p_table.step, "ROUND PLAYED")
board = SmartX(p_table, debug=True)

count = {'O': 0, 'X': 0, '=': 0}
for step in range(1000):
    (result_board, winner) = board.play_game()
    print("WINNER", winner)
    board.print_board(result_board)
    count[winner] += 1
    p_table.next_step()
    if winner == 'O':
        sys.exit(0)

# p_table.save('p_table.dat')
print(count)

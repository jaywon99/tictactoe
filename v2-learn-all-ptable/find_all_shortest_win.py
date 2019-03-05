import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from tictactoe import TicTacToeBoard

def find_next(t):
    # print("TRACE", t.seq)
    winner = t.is_win()
    if winner != ' ':
        print(t.seq, winner, t.get_board_id())
        return

    candidates = t.get_candidates()
    ids = {}
    for c in candidates[::-1]:
        ids[t.get_board_id(t.seq+str(c))] = c
    candidates = list(ids.values())
    candidates.sort()

    for candidate in candidates:
        t.play(candidate)
        winner = t.is_win()
        if winner != ' ':
            print(t.seq, winner, t.get_board_id())
            t.unplay()
            return
        t.unplay()

    for candidate in candidates:
        t.play(candidate)
        find_next(t)
        t.unplay()


t = TicTacToeBoard()
t.init_board()
find_next(t)


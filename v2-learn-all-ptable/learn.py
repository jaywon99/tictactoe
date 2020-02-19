import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

import tictactoe.gym as gym
import tictactoe.agent as agent
import tictactoe.utils as utils
from ptable_agent import SmartAgent

from ptable import PredictionTable

p_table = PredictionTable(learning_rate=0.5)
p_table.load('tictactoe.dat')

player1 = SmartAgent(p_table, debug=False)
player2 = SmartAgent(p_table, debug=False)

env = gym.getEnv()
dual = agent.DualAgent(player1, player2)

dual_x = agent.DualAgent(player1, agent.RandomAgent())
dual_o = agent.DualAgent(agent.RandomAgent(), player2)

with open('allpath.txt', 'r') as f:
    history = f.read().split('\n')[:-1]

EPOCH = 100

for epoch in range(0, EPOCH, 10):

    print("EPOCH", epoch)
    dual.set_train_mode(True)
    for step in range(10):
        p_table.next_step()

        random.shuffle(history)
        for line in history:
            # print(line.strip(), line.strip().split(' '))
            seq, winner, _ = line.strip().split(' ')
            seq1 = [int(s) for i, s in enumerate(seq) if i%2==0]
            seq2 = [int(s) for i, s in enumerate(seq) if i%2==1]

            player1.set_queue(seq1)
            player2.set_queue(seq2)
            winner = utils.play(env, dual)

    p_table.save('tictactoe.dat')

    dual.set_train_mode(False)
    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(1000):
        winner = utils.play(env, dual_o)
        count[winner] += 1
    print("STEP", step, "SMART_O", count)

    count = {-1: 0, 1: 0, 0: 0}
    for step1 in range(1000):
        winner = utils.play(env, dual_x)
        count[winner] += 1
    print("STEP", step, "SMART_X", count)

    winner = utils.play(env, dual)
    print("STEP", step, "WINNER", winner)
    if winner != 0:
        env.render()



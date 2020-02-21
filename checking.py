'''
Main checking start up point by learning algorithm
'''

import sys

import tictactoe.gym as gym
from tictactoe.negamax.negamax import NegamaxAgent

def usage():
    '''
    print usage and exit
    '''
    print("Usage: checking.py algorithm")
    print("\talgorithm: ptable, qlearning, dqn, ddqn, pgn")
    sys.exit(-1)

def main(learn):
    ''' call learning function with args '''
    env = gym.getEnv()  # Game Environment
    best_player = NegamaxAgent(env) # Best Player Algorithm

    learn(env, best_player)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()

    if sys.argv[1] == "ptable":
        import ptable as learning_alg
        main(learning_alg.checking)
    elif sys.argv[1] == "qlearning":
        import qlearning as learning_alg
        main(learning_alg.checking)
    elif sys.argv[1] == "dqn":
        import dqn as learning_alg
        main(learning_alg.checking)
    elif sys.argv[1] == "ddqn":
        pass
    elif sys.argv[1] == "pgn":
        pass
    else:
        usage()

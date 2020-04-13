import sys
import random

from boardAI import RandomPlayer, Arena, PlayerMode
from tictactoe import TicTacToeBoard
from players import load_player

board = TicTacToeBoard()
players = []
players.append(load_player(name="RandomPlayer1", storage="models/random_player1", cls="RandomPlayer"))
players.append(load_player(name="PTablePlayer1", storage="models/ptable_player1", cls='PredictionTablePlayer'))
players.append(load_player(name="QPlayer1", storage="models/q_player1", cls="QLearningPlayer"))
players.append(load_player(name="DQNPlayer1", storage="models/dqn_player1", cls="DQNPlayer", network_storage='./models/dqn1.ckpt'))
players.append(load_player(name="DDQNPlayer1", storage="models/ddqn_player1", cls="DDQNPlayer", network_storage='./models/ddqn1.ckpt'))
players.append(load_player(name="NegamaxPlayer", storage="models/negamax_player1", cls="NegamaxPlayer"))
players.append(load_player(name="MCTSRandomPlayer", storage="models/mcts_random_player1", cls="MCTSRandomPlayer"))

# arena_train.reset()
# arena_train.duel(render='human')

# arena_play.reset()
# arena_play.duel(render='human')

for i in range(1000):
    for _ in range(100):        
        arena_train = Arena(board, random.sample(players, 2), mode = PlayerMode.TRAIN)
        arena_train.reset()
        arena_train.duel(render=None)

    for _ in range(100):
        arena_play = Arena(board, random.sample(players, 2), mode = PlayerMode.PLAY)
        arena_play.reset()
        arena_play.duel(render=None)
    print("PLAY", i*100, [str(p.elo) for p in players])

    for player in players:
        player.save()


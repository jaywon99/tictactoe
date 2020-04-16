from boardAI import Arena, PlayerMode
from tictactoe import TicTacToeBoard
from players import load_player

board = TicTacToeBoard()
# player1 = load_player(name="RandomPlayer1",
#                       storage="models/random_playerX", cls="RandomPlayer")
player1 = load_player(name="RandomPlusPlayer1",
                      storage="models/randomplus_playerX", cls="RandomPlusPlayer")
# player1 = load_player(name="PTablePlayer1",
#                       storage="models/ptable_playerX", cls='PredictionTablePlayer')
# player1 = load_player(
#     name="QPlayerX", storage="models/q_playerX", cls="QLearningPlayer")
# player1 = load_player(name="DQNPlayerX", storage="models/dqn_playerX",
#                       cls="DQNPlayer", network_storage='./models/dqn1.ckpt')
# player1 = load_player(name="DDQNPlayerX", storage="models/ddqn_playerX",
#                       cls="DDQNPlayer", network_storage='./models/ddqn1.ckpt')
# player1 = load_player(
#     name="Human1", storage="models/human_player1", cls="HumanPlayer")
# player1 = load_player(name="NegamaxPlayer1",
#                       storage="models/negamax_player1", cls="NegamaxPlayer")
# player1 = load_player(name="MCTSRandomPlayer1",
#                       storage="models/mcts_random_player1", cls="MCTSRandomPlayer")
# player2 = load_player(name="RandomPlayer2",
#                       storage="models/random_player2", cls="RandomPlayer")
# player2 = load_player(name="RandomPlusPlayer2",
#                       storage="models/randomplus_player2", cls="RandomPlusPlayer")
# player2 = load_player(name="PTablePlayer2",
#                       storage="models/ptable_player2", cls='PredictionTablePlayer')
# player2 = load_player(
#     name="QPlayer2", storage="models/q_player2", cls="QLearningPlayer")
# player2 = load_player(name="DQNPlayer2", storage="models/dqn_player2",
#                       cls="DQNPlayer", network_storage='./models/dqn2.ckpt')
# player2 = load_player(name="DDQNPlayer2", storage="models/ddqn_player2",
#                       cls="DDQNPlayer", network_storage='./models/ddqn2.ckpt')
# player2 = load_player(
#     name="Human2", storage="models/human_player2", cls="HumanPlayer")
# player2 = load_player(name="NegamaxPlayer2",
#                       storage="models/negamax_player2", cls="NegamaxPlayer")
player2 = load_player(name="MCTSRandomPlayer2",
                      storage="models/mcts_random_player2", cls="MCTSRandomPlayer")

arena_train = Arena(board, [player1, player2], mode=PlayerMode.TRAIN)
arena_play = Arena(board, [player1, player2], mode=PlayerMode.PLAY)

# arena_train.reset()
# arena_train.duel(render='human')

# arena_play.reset()
# arena_play.duel(render='human')
# print(arena_play.results, player1.elo, player2.elo)

print(player1.elo, player2.elo)
for _ in range(100):
    for i in range(1000):
        arena_train.reset()
        arena_train.duel(render=None)

    arena_play.reset_results()
    for i in range(100):
        arena_play.reset()
        arena_play.duel(render=None)
    print(arena_play.results, player1.elo, player2.elo)

    player1.save()
    player2.save()

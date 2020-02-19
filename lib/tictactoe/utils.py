def board_to_id(board):
    _id = 0
    for digit in (board):
        _id = (_id << 2) | (digit & 3)
    return _id

# TOTAL 8 converion way
CONVERSIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], # -
    [0, 3, 6, 1, 4, 7, 2, 5, 8], # >y  (rotate 90 and flip y)
    [2, 1, 0, 5, 4, 3, 8, 7, 6], # y   (flip y)
    [2, 5, 8, 1, 4, 7, 0, 3, 6], # >>> (rotate 90 3 times)
    [6, 3, 0, 7, 4, 1, 8, 5, 2], # >   (rotate 90 1 time)
    [6, 7, 8, 3, 4, 5, 0, 1, 2], # >>y (rotate 90 2 times and flip y)
    [8, 5, 2, 7, 4, 1, 6, 3, 0], # >>>y (rotate 90 3 times and flip y)
    [8, 7, 6, 5, 4, 3, 2, 1, 0]  # >>  (rotate 90 2 times)
]

class OptimalBoard:
    def __init__(self, board):
        self.idx = -1
        self.board_id = 100000000
        for idx, C in enumerate(CONVERSIONS):
            y = [0] * 9
            for i, v in enumerate(board):
                y[C[i]] = v
            _id = board_to_id(y)
            if _id < self.board_id:
                self.board_id = _id
                self.idx = idx
                self.optimized_board = y

    def get_board_id(self):
        return self.board_id

    def convert_available_actions(self, available_actions):
        C = CONVERSIONS[self.idx]
        return [C[action] for action in available_actions]

    def convert_available_action(self, action):
        C = CONVERSIONS[self.idx]
        return C[action]

    def convert_to_original_action(self, action):
        C = CONVERSIONS[self.idx]
        for i, v in enumerate(C):
            if v == action:
                return i
        assert 0 == 1

def play(env, dual, render = False, recorded = False):
    next_state = env.reset()
    dual.reset()
    done = False
    action = None
    reward = 0
    history = []
    while not done:
        if render:
            env.render()
        agent = dual.next_agent()
        if agent.is_played():
            agent.feedback(next_state, reward, False)   # generally feedback is coming after other player played.

        action = agent.next_action(next_state, env.available_actions())
        (next_state, reward, done, _) = env.step(action)
        if recorded:
            history.append(action)

    if render:
        env.render()

    if reward == 0: # means TIE
        dual.current_agent().feedback(next_state, 0, True)
        dual.other_agent().feedback(next_state, 0, True)
    else: # means current_agent WIN
        dual.current_agent().feedback(next_state, 1, True)
        dual.other_agent().feedback(next_state, -1, True)
    dual.current_agent().episode_feedback(1)
    dual.other_agent().episode_feedback(-1)

    if recorded:
        return (reward, history)
    return reward

def test_traverse(env, agent, next_state, seq):
    scores = {'p': 0, 't': 0, 'o': 0}
    actions = env.available_actions()
    for action in actions:
        memento = env.create_memento()
        (x, reward, done, _) = env.step(action)
        # env.render()
        if done: 
            if reward == 0:
                scores['t'] += 1
            else:
                scores['o'] += 1
        else:
            new_scores = do_player1_test(env, agent, x, seq+str(action))
            scores['p'] += new_scores['p']
            scores['t'] += new_scores['t']
            scores['o'] += new_scores['o']
        env.set_memento(memento)

    return scores

def do_player1_test(env, agent, next_state, seq):
    actions = env.available_actions()

    action = agent.next_action(next_state, actions)
    (next_state, reward, done, _) = env.step(action)
    # env.render()
    if done: 
        if reward == 0:
            return {'p': 0, 't': 1, 'o': 0}
        else:
            return {'p': 1, 't': 0, 'o': 0}

    return test_traverse(env, agent, next_state, seq+str(action))

def test_player1(env, agent):
    next_state = env.reset()
    agent.reset(False, -1)
    done = False

    return do_player1_test(env, agent, next_state, "")

def test_player2(env, agent):
    next_state = env.reset()
    agent.reset(False, -1)
    done = False

    return test_traverse(env, agent, next_state, "")



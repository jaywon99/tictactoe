
def compact_observation(obs):
    _id = 0
    _id = (obs[1] & 3) # 거의 필요없음.
    for digit in (obs[0]):
        _id = (_id << 2) | (digit & 3)
    return _id

def play(env, dual, render = False):
    next_state = env.reset()
    dual.reset()
    done = False
    while not done:
        if render:
            env.render()
        agent = dual.next_agent()
        action = agent.next_action(next_state, env.available_actions())
        (next_state, reward, done, _) = env.step(action)

    if render:
        env.render()

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
    agent.reset()
    done = False

    return do_player1_test(env, agent, next_state, "")

def test_player2(env, agent):
    next_state = env.reset()
    agent.reset()
    done = False

    return test_traverse(env, agent, next_state, "")




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
        agent = dual.next_agent()
        action = agent.next_action(next_state, env.available_actions())
        (next_state, reward, done, _) = env.step(action)
        if render:
            env.render()

    return reward


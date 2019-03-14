import random


def gym_test():
    '''
    Example of using `gym.make` to construct a registered environment.
    '''
    import gym
    import gymberkeleyrl
    env = gym.make("Pacman-MediumClassic-v0")
    observation = env.reset()
    for _ in range(1000):
        actions = env.getPossibleActions()
        print('obs:', observation)
        print('actions:', actions)
        print('agentIndex:', env.agent_idx)
        action = random.choice(actions) # env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            done = False
            observation = env.reset()


if __name__ == '__main__':
    gym_test()
#     main()

import random
import gym
import gymberkeleyrl
from gymberkeleyrl.envs import PacmanEnv


def gym_test():
    '''
    Example of using `gym.make` to construct a registered environment.
    '''
#     env = gym.make("Pacman-MediumClassic-v0")
    env = PacmanEnv(layout='minimaxClassic')
    observation = env.reset()
    for _ in range(100):
        actions = env.getPossibleActions()
        action = random.choice(actions) # env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            done = False
            observation = env.reset()


if __name__ == '__main__':
    gym_test()
#     main()

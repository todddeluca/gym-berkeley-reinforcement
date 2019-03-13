from gym.envs.registration import register

register(
    id='pacman-smallgrid-v0',
    entry_point='gymberkeleyrl.envs:PacmanEnv',
)

register(
    id='gridworld-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'length': 2}
)


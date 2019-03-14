from gym.envs.registration import register

register(
    id='Gridworld-BookGrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'BookGrid'}
)

register(
    id='Gridworld-BridgeGrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'BridgeGrid'}
)

register(
    id='Gridworld-CliffGrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'CliffGrid'}
)

register(
    id='Gridworld-MazeGrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'MazeGrid'}
)

register(
    id='Pacman-MediumClassic-v0',
    entry_point='gymberkeleyrl.envs:PacmanEnv',
    kwargs={'layout': 'mediumClassic'}
)


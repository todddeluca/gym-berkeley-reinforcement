from gym.envs.registration import register

# Gridworld Environments

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

# Pacman Environments

register(
    id='Pacman-MediumClassic-v0',
    entry_point='gymberkeleyrl.envs:PacmanEnv',
    kwargs={'layout': 'mediumClassic'}
)

register(
    id='Pacman-MinimaxClassic-v0',
    entry_point='gymberkeleyrl.envs:PacmanEnv',
    kwargs={'layout': 'minimaxClassic'}
)

register(
    id='Pacman-SmallGrid-v0',
    entry_point='gymberkeleyrl.envs:PacmanEnv',
    kwargs={'layout': 'smallGrid'}
)



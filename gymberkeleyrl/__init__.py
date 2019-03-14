from gym.envs.registration import register

register(
    id='gridworld-bookgrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'BookGrid'}
)

register(
    id='gridworld-bridgegrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'BridgeGrid'}
)

register(
    id='gridworld-cliffgrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'CliffGrid'}
)

register(
    id='gridworld-mazegrid-v0',
    entry_point='gymberkeleyrl.envs:GridworldEnv',
    kwargs={'grid': 'MazeGrid'}
)


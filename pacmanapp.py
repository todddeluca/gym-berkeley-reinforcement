
import argparse
import numpy as np
import pickle
import random
import time
import gym
import gymberkeleyrl
from gymberkeleyrl.envs import PacmanEnv

from reinforcement.pacman import parseAgentArgs, replayGame
from reinforcement import (ghostAgents, learningAgents, qlearningAgents, 
                           keyboardAgents, pacmanAgents, valueIterationAgents)

agent_modules = [ghostAgents, learningAgents, qlearningAgents, 
                 keyboardAgents, pacmanAgents, valueIterationAgents]


def loadAgent(pacman, nographics):
    '''
    Return the agent class constructor from one of the agent modules.
    '''
    for module in agent_modules:
        if pacman in dir(module):
            if nographics and 'keyboardAgents' in module.__name__:
                raise Exception(
                    'Using the keyboard requires graphics (not text display)')
            return getattr(module, pacman)
        
    raise Exception('The agent ' + pacman +' is not specified in any *Agents.py.')
    
    
def default(str):
    return str + ' [Default: %(default)s]'


def parseArgs():
    usage = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-n', '--numGames', type=int,
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_argument('-l', '--layout', 
                      help=default(
                          'the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_argument('-p', '--pacman',
                      help=default(
                          'the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_argument('-t', '--textGraphics', action='store_true',
                      help='Display output as text only', default=False)
    parser.add_argument('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_argument('-g', '--ghosts', dest='ghost',
                      help=default(
                          'the ghost agent TYPE in the ghostAgents module to use'),
                      metavar='TYPE', default='RandomGhost')
    parser.add_argument('-k', '--numghosts', type=int, dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_argument('-z', '--zoom', type=float, 
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_argument('-f', '--fixRandomSeed', action='store_true', 
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_argument('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_argument('--replay', dest='gameToReplay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_argument('-a', '--agentArgs', 
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_argument('-x', '--numTraining',  type=int,
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_argument('--frameTime', type=float,
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_argument('-c', '--catchExceptions', action='store_true',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_argument('--timeout', type=int,
                      help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

    args = parser.parse_args()
    return args


def run_game(env, agents, quiet, initialize_display):
    
    # keyboard agent requires display to be initialized at the start of the game
    state = env.reset(quiet=quiet, initialize_display=initialize_display)
    done = False
    agent_idx = 0

    # Alternative calculating reward from state scores.
    # track rewards of all agents, so agent can calculate the cumulative reward
    # since their previous action.
    agent_rewards = np.zeros(len(agents))

    # Initialize agents
    for agent in agents:
        if "registerInitialState" in dir(agent):
            agent.registerInitialState(state)

    while not done:
        agent = agents[agent_idx]

        # Pacman ReinforcementAgent classes update by using the 
        # score from state and the last state the agent saw, to calculate the reward.
#             if 'observationFunction' in dir(agent):
#                 agent.observationFunction(state)

        actions = env.getPossibleActions()
        # action = random.choice(actions) # env.action_space.sample() # your agent here (this takes random actions)
        action = agent.getAction(state)

        # reward is not cummulative since the last time this agent acted
        next_state, reward, done, info = env.step(action)
        agent_rewards[agent_idx] = reward

        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, next_state, agent_rewards.sum())

        if not quiet:
            env.render()

        state = next_state
        agent_idx = (agent_idx + 1) % len(agents)

    for agent in agents:
        if "final" in dir(agent):
            agent.final(state)

    # for recording game
    game = info['game']
    layout = info['layout']
    return game, layout


def main():
    
    args = parseArgs()

    # make environment
    env = PacmanEnv(layout=args.layout, max_ghosts=args.numGhosts, catch_exceptions=args.catchExceptions,
                    timeout=args.timeout, quiet_graphics=args.quietGraphics,
                    text_graphics=args.textGraphics, frame_time=args.frameTime, zoom=args.zoom,
                    fix_random_seed=args.fixRandomSeed)
    num_ghosts = env.num_ghosts

    # make actionFn
    def make_actionFn(agent_index):
        def actionFn(state):
            return env.getPossibleActions(state, agent_index)
        
    noKeyboard = args.gameToReplay == None and (
        args.textGraphics or args.quietGraphics)
    
    # make Pacman agent
    pacmanType = loadAgent(args.pacman, noKeyboard)
    agentOpts = parseAgentArgs(args.agentArgs)
    if args.numTraining > 0:
        if 'numTraining' not in agentOpts:
            agentOpts['numTraining'] = args.numTraining
            
    # Do not pass actionFn to agents that won't take it.
    if args.pacman not in ['KeyboardAgent', 'LeftTurnAgent', 'GreedyAgent']:
        agentOpts['actionFn'] = make_actionFn(0) # pacman is the first agent
        
    pacman = pacmanType(**agentOpts)

    # Don't display training games
    # is this ever used?
    if 'numTrain' in agentOpts:
        numQuiet = int(agentOpts['numTrain'])
        numIgnore = int(agentOpts['numTrain'])
    
    # make ghost agents
    ghostType = loadAgent(args.ghost, noKeyboard)
    ghosts = [ghostType(i+1) for i in range(num_ghosts)]
    
    agents = [pacman] + ghosts

    # Special case: recorded games
    if args.gameToReplay != None:
        print('Replaying recorded game %s.' % args.gameToReplay)
        with open(args.gameToReplay, 'rb') as f:
            recorded = pickle.load(f)
            recorded['display'] = env.display
            replayGame(**recorded)
            sys.exit(0)

    
    # run games
    games = []
    for i in range(args.numGames):
        quiet = i < args.numTraining # shush training games
        initialize_display = args.pacman == 'KeyboardAgent' # start display early to get keystrokes
        game, layout = run_game(env, agents, quiet, initialize_display=initialize_display)
        if not quiet:
            games.append(game)

        if args.record:
            fname = ('recorded-game-%d' % (i + 1)) + \
                '-'.join([str(t) for t in time.localtime()[1:6]])
            with open(fname, 'wb') as f:
                components = {'layout': layout, 'actions': game.moveHistory}
                pickle.dump(components, f)

    # print testing stats
    if (args.numGames - args.numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' %
              (wins.count(True), len(wins), winRate))
        print('Record:       ', ', '.join(
            [['Loss', 'Win'][int(w)] for w in wins]))

    return games


def gym_test():
    '''
    Example of using `gym.make` to construct a registered environment.
    '''
    env = gym.make("Pacman-MinimaxClassic-v0")
    num_agents = env.num_agents # get number of ghosts + 1 from env
    agents = [pacmanAgents.GreedyAgent()]
    agents += [ghostAgents.RandomGhost(i) for i in range(1, num_agents)]
    num_games = 2
    for j in range(2): # num games
        state = env.reset()
        done = False
        agent_idx = 0
        agent_rewards = np.zeros(num_agents)
        while not done:
            agent = agents[agent_idx]
#             actions = env.getPossibleActions()
#             action = random.choice(actions) 
            action = agent.getAction(state)
            next_state, reward, done, info = env.step(action)
            agent_rewards[agent_idx] = reward
            if agent_idx == 0:
                print('Rewards since Pacman\'s last move:', agent_rewards.sum())
            env.render()
            state = next_state
            agent_idx = (agent_idx + 1) % num_agents


if __name__ == '__main__':
#     gym_test()
    main()

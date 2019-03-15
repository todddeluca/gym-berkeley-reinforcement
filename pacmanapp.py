
import argparse
import random
import gym
import gymberkeleyrl
from gymberkeleyrl.envs import PacmanEnv

from reinforcement.pacman import parseAgentArgs
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



def main():
    
    args = parseArgs()

    # make environment
    env = PacmanEnv(layout=args.layout)    
    num_ghosts = env.num_ghosts

    # make actionFn
    def actionFn(state):
        return env.getPossibleActions(state)
        
    # make Pacman agent
    noKeyboard = args.gameToReplay == None and (
        args.textGraphics or args.quietGraphics)
    pacmanType = loadAgent(args.pacman, noKeyboard)
    agentOpts = parseAgentArgs(args.agentArgs)
    # Do not pass actionFn to agents that won't take it.
    if args.pacman not in ['KeyboardAgent', 'LeftTurnAgent', 'GreedyAgent']:
        agentOpts['actionFn'] = actionFn
        
    pacman = pacmanType(**agentOpts)

    # make ghost agents
    ghostType = loadAgent(args.ghost, noKeyboard)
    ghosts = [ghostType(i+1) for i in range(num_ghosts)]
    
    agents = [pacman] + ghosts
    
    # run games
    for i in range(args.numGames):
        # keyboard agent requires display to be initialized at the start of the game
        state = env.reset(initialize_display=(args.pacman == 'KeyboardAgent'))
        done = False
        agent_idx = 0
        for agent in agents:
            if "registerInitialState" in dir(agent):
                agent.registerInitialState(state)
                
        while not done:
            agent = agents[agent_idx]
            if 'observationFunction' in dir(agent):
                agent.observationFunction(state)
                
            actions = env.getPossibleActions()
            # action = random.choice(actions) # env.action_space.sample() # your agent here (this takes random actions)
            action = agent.getAction(state)
            
            # reward is not cummulative since the last time this agent acted
            next_state, reward, done, info = env.step(action)
            
            # oh no agent_idx has changed, so actionFn is broken.
            # if 'observeTransition' in dir(agent):
            #     agent.observeTransition(state, action, next_state, reward)

            if i >= args.numTraining:
                env.render()
                
            state = next_state
            agent_idx = (agent_idx + 1) % len(agents)

        for agent in agents:
            if "final" in dir(agent):
                agent.final(state)

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
#     gym_test()
    main()

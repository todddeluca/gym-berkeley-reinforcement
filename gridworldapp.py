
import argparse
from gymberkeleyrl.envs.gridworldenv import GridworldEnv
from reinforcement import valueIterationAgents, qlearningAgents
import random


class RandomAgent:
    '''
    An agent that acts randomly.
    '''
    def __init__(self, actionFn):
        self.actionFn = actionFn
        
    def getAction(self, state):
        return random.choice(self.actionFn(state))
    
    def getValue(self, state):
        return 0.0
    
    def getQValue(self, state, action):
        return 0.0
    
    def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
    
    def update(self, state, action, nextState, reward):
        pass


class UserAgent:
    '''
    Get an action from the user.

    Used for debugging and lecture demos.
    '''
    def __init__(self, actionFn):
        self.actionFn = actionFn
        
    def getAction(self, state):
        from gymberkeleyrl.reinforcement import graphicsUtils
        action = None
        while True:
            keys = graphicsUtils.wait_for_keys()
            if 'Up' in keys: action = 'north'
            if 'Down' in keys: action = 'south'
            if 'Left' in keys: action = 'west'
            if 'Right' in keys: action = 'east'
            if 'q' in keys: sys.exit(0)
            if action == None: continue
            break
            
        actions = self.actionFn(state)
        if action not in actions:
            action = actions[0]
            
        return action
    
    def getValue(self, state):
        return 0.0
    
    def getQValue(self, state, action):
        return 0.0
    
    def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
    
    def update(self, state, action, nextState, reward):
        pass


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--livingReward', type=float, default=0.0,
                         metavar='R', help='Reward for living for a time step (default %(default)s)')
    parser.add_argument('-d', '--discount',
                         type=float, default=0.9,
                         help='Discount on future (default %(default)s)')
    parser.add_argument('-n', '--noise',
                         type=float, default=0.2,
                         metavar="P", help='How often action results in unintended direction (default %(default)s)' )
    parser.add_argument('-e', '--epsilon',
                         type=float, default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %(default)s)')
    parser.add_argument('-l', '--learningRate',
                         type=float, default=0.5,
                         metavar="P", help='TD learning rate (default %(default)s)' )
    parser.add_argument('-i', '--iterations',
                         type=int, dest='iters', default=10,
                         metavar="K", help='Number of rounds of value iteration (default %(default)s)')
    parser.add_argument('-k', '--episodes',
                         type=int, default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %(default)s)')
    parser.add_argument('-g', '--grid',
                         metavar="G", default="BookGrid", choices=['BookGrid', 'BridgeGrid', 'CliffGrid', 'MazeGrid'],
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %(default)s)' )
    parser.add_argument('-w', '--windowSize', metavar="X", type=int, dest='gridSize', default=150,
                         help='Request a window width of X pixels *per grid cell* (default %(default)s)')
    parser.add_argument('-a', '--agent', metavar="A",
                         default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %(default)s)')
    parser.add_argument('-t', '--text', action='store_true',
                         dest='textDisplay', default=False,
                         help='Use text-only ASCII display')
    parser.add_argument('-p', '--pause', action='store_true',
                         default=False,
                         help='Pause GUI after each time step when running the MDP')
    parser.add_argument('-q', '--quiet',action='store_true',
                         default=False,
                         help='Skip display of any learning episodes')
    parser.add_argument('-s', '--speed', metavar="S", type=float,
                         default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %(default)s)')
    parser.add_argument('-m', '--manual',action='store_true',
                         default=False,
                         help='Manually control agent')
    parser.add_argument('-v', '--valueSteps', action='store_true', default=False,
                         help='Display each step of value iteration')
    args = parser.parse_args()
    # MANAGE CONFLICTS
    if args.textDisplay or args.quiet:
        args.pause = False

    if args.manual:
        args.pause = True
        
    return args


def main():
    args = parseArgs()
    
    # Make environment
    env = GridworldEnv(args.grid, args.livingReward, args.noise, args.textDisplay,
                       args.gridSize, args.speed, args.pause)
    
    
    # Make actionFn
    def actionFn(state):
        return env.getPossibleActions(state)
            
    # Make agent
    agent = None
    if args.agent == 'value':
        agent = valueIterationAgents.ValueIterationAgent(env.mdp, args.discount, args.iters)
    elif args.agent == 'q':
        qLearnOpts = {'gamma': args.discount,
                      'alpha': args.learningRate,
                      'epsilon': args.epsilon,
                      'actionFn': actionFn}
        agent = qlearningAgents.QLearningAgent(**qLearnOpts)
    elif args.agent == 'random':
        # # No reason to use the random agent without episodes
        if args.episodes == 0:
            args.episodes = 10
            
        agent = RandomAgent(actionFn=actionFn)
    else:
        if args.manual:
            agent = UserAgent(actionFn=actionFn)
        else:
            raise Exception('Unknown agent type: '+args.agent)

    if args.manual and args.agent == 'q':
        agent.getAction = UserAgent(actionFn=actionFn).getAction
        
    if args.quiet:
        message = lambda x: None
    else:
        message = lambda x: print(x)
        
    # Display Value Iterations for the Value Agent
    try:
        if not args.manual and args.agent == 'value':
            if args.valueSteps:
                for i in range(args.iters):
                    temp_agent = valueIterationAgents.ValueIterationAgent(env.mdp, args.discount, i)
                    env.display.displayValues(temp_agent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
                    env.display.pause()

            env.display.displayValues(agent, message = "VALUES AFTER "+str(args.iters)+" ITERATIONS")
            env.display.pause()
            env.display.displayQValues(agent, message = "Q-VALUES AFTER "+str(args.iters)+" ITERATIONS")
            env.display.pause()
    except KeyboardInterrupt:
        sys.exit(0)
    
    # Run Episodes
    if args.episodes > 0:
        print()
        print(("RUNNING", args.episodes, "EPISODES"))
        print()
        
    returns = 0
    for episode in range(1, args.episodes + 1):
        if 'startEpisode' in dir(agent): 
            agent.startEpisode()
            
        message("BEGINNING EPISODE: "+str(episode)+"\n")
        curr_state = env.reset()
        done = False
        episode_returns = 0
        total_discount = 1
        while not done:
            env.render(agent=agent)
            action = agent.getAction(curr_state)
            if action == None:
                raise Exception('Error: Agent returned None action')

            next_state, reward, done, info = env.step(action)
            print(next_state, reward, done, info)
            message("Started in state: "+str(curr_state)+
                    "\nTook action: "+str(action)+
                    "\nEnded in state: "+str(next_state)+
                    "\nGot reward: "+str(reward)+"\n")
            
            # UPDATE LEARNER
            if 'observeTransition' in dir(agent):
                agent.observeTransition(curr_state, action, next_state, reward)

            curr_state = next_state
            
            episode_returns += reward * total_discount
            total_discount *= args.discount
            if done:
                message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(episode_returns)+"\n")
                returns += episode_returns

        if 'stopEpisode' in dir(agent):
            agent.stopEpisode()
        
    if args.episodes > 0:
        print()
        print(("AVERAGE RETURNS FROM START STATE: "+str((returns + 0.0) / args.episodes)))
        print()
        print()

    # DISPLAY POST-LEARNING VALUES / Q-VALUES
    if args.agent == 'q' and not args.manual:
        try:
            env.display.displayQValues(agent, message = "Q-VALUES AFTER "+str(args.episodes)+" EPISODES")
            env.display.pause()
            env.display.displayValues(agent, message = "VALUES AFTER "+str(args.episodes)+" EPISODES")
            env.display.pause()
        except KeyboardInterrupt:
            sys.exit(0)


def gym_test():
    '''
    Example of using `gym.make` to construct a registered environment.
    '''
    import gym
    import gymberkeleyrl
    env = gym.make("gridworld-mazegrid-v0")
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        action = 'exit'
        observation, reward, done, info = env.step(action)
        if done:
            done = False
            observation = env.reset()


if __name__ == '__main__':
#     gym_test()
    main()

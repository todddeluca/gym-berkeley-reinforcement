import gym
from gym import spaces
from gym.spaces import Space
from gym.utils import seeding

import numpy as np
from gymberkeleyrl.reinforcement import gridworld
from gymberkeleyrl.reinforcement import environment
from gymberkeleyrl.reinforcement import textGridworldDisplay
from gymberkeleyrl.reinforcement import graphicsGridworldDisplay
from gymberkeleyrl.spaces import ObjectSpace


# Useful references for creating a custom gym environment
# gym.Env: https://github.com/openai/gym/blob/master/gym/core.py
# Example environment -- soccer: https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
# https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai
# Gym README: https://github.com/openai/gym/
# Gym Docs: https://gym.openai.com/docs/


class StubDisplayAgent:
    '''
    render() expects an agent with certain functions
    '''
    # expected by display.displayQValues
    def getQValue(*args):
        return 0
    
    # expected by display.displayValues
    def getValue(*args):
        return 0

    # expected by display.displayValues
    def getPolicy(*args):
        return None

    
class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid, livingReward, noise, textDisplay, gridSize, speed, pause):
        '''
        '''
        self.pause = pause
        
        # initialize mdp and env (code from gridworld.py).
        mdpFunction = getattr(gridworld, "get" + grid)
        self.mdp = mdpFunction() # Used by dynamic programming ValueIterationAgent
        self.mdp.setLivingReward(livingReward)
        self.mdp.setNoise(noise)
        self.env = gridworld.GridworldEnvironment(self.mdp)
        self.state = self.env.getCurrentState()

        # initialize display
        self.display = textGridworldDisplay.TextGridworldDisplay(self.mdp)
        if not textDisplay:
            self.display = graphicsGridworldDisplay.GraphicsGridworldDisplay(self.mdp, gridSize, speed)
        try:
            self.display.start()
        except KeyboardInterrupt:
            sys.exit(0)

        # legal actions: (north, east, south, west) or (exit,) or (,)
        self.action_space = ObjectSpace(self.env.getPossibleActions(self.state))
        
        # observation is a state, one of the possible states of the mdp
        self.observation_space = ObjectSpace(self.mdp.getStates())

        self.seed()
        self.reset()
        
    def reset(self):
        '''
        Reset the environment state. Return initial observation.
        '''
        self.env.reset()
        self.state = self.env.getCurrentState()
        self.action_space = ObjectSpace(self.env.getPossibleActions(self.state))
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        next_state, reward = self.env.doAction(action)
        self.state = next_state
        self.action_space = ObjectSpace(self.env.getPossibleActions(self.state))
        done = (len(self.action_space) == 0) # done if no legal actions
        
        return (next_state, reward, done, {})

    def render(self, mode='human', agent=None):
        '''
        For humans, visualize something. For others, return data.
        '''
        if agent is None:
            agent = StubDisplayAgent()
            
        if 'getQValue' in dir(agent):
            self.display.displayQValues(agent, self.state, "CURRENT Q-VALUES")
        else:
            self.display.displayValues(agent, self.state, "CURRENT VALUES")

        if self.pause:
            self.display.pause()

    def getPossibleActions(self, state):
        '''
        To be consistent with the berkeley agent interface, 
        which expects a function that takes a state and returns a
        list of possible actions, this function is added.
        
        Other agents can use the action_space.objects attribute to get the
        possible actions for the current environment state.
        '''
        return self.env.getPossibleActions(state)
        
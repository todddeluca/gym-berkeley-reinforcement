

import gym
from gym import spaces
from gym.spaces import Space
from gym.utils import seeding

import numpy as np
import random

from reinforcement import game
from reinforcement.game import Agent
from reinforcement import layout as layout_
from reinforcement import pacman
from reinforcement import textDisplay
from reinforcement.pacman import ClassicGameRules

from gymberkeleyrl.spaces import ObjectSpace


class PacmanEnv(gym.Env):
    '''
    Pacman is a multi-agent game: one pacman agent and a configurable number of ghost agents.
    Each agent takes turns playing one move, first pacman, then the ghosts. The state updates
    after each move. The reward is the difference in score, so the reward is from the
    perspective of pacman, not the ghosts.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, layout='mediumClassic', max_ghosts=4, catch_exceptions=False, timeout=30, 
                 quiet_graphics=False, text_graphics=False, frame_time=0.1, zoom=1.0, 
                 fix_random_seed=False):
        '''
        Number of ghosts is min(max_ghosts, layout.getNumGhosts()).
        '''
        self.agent_idx = 0 # tracks the index of the next agent to play.
        self.catch_exceptions = catch_exceptions
        self.max_ghosts = max_ghosts
        
        if fix_random_seed:
            random.seed('cs188')

        # Make layout
        self.layout = layout_.getLayout(layout)
        if self.layout is None:
            raise Exception("The layout " + layout + " cannot be found")
            
        self.num_ghosts = min(max_ghosts, self.layout.getNumGhosts())
        self.num_agents = self.num_ghosts + 1
                
        # Make display
        self.display_initialized = False
        self.null_display = textDisplay.NullGraphics()
        if quiet_graphics:
            self.display = null_display
        elif text_graphics:
            textDisplay.SLEEP_TIME = frame_time
            self.display = textDisplay.PacmanGraphics()
        else:
            from reinforcement import graphicsDisplay
            self.display = graphicsDisplay.PacmanGraphics(zoom, frameTime=frame_time)
        
        # Make rules
        self.rules = ClassicGameRules(timeout)
        self.game = None
        
        # Actions depend on agent and state
        self.action_space = None
        
        # A state/observation is a rather complex object.
        self.observation_space = None

        self.seed()
        
    def reset(self, quiet=False):
        '''
        Reset the environment state. Return initial observation.
        '''
        if self.display_initialized:
            self.display.finish()
            self.display_initialized = False

        # game is passed dummy pacman and ghost agents, since the game
        # is not being used for the game.run().
        self.game = self.rules.newGame(
            self.layout, Agent(0), [Agent(i) for i in range(1, self.max_ghosts + 1)],
            self.display, quiet, self.catch_exceptions)
        self.agent_idx = 0 # pacman moves first

        return self.game.state

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
        # Execute the action
        self.game.moveHistory.append((self.agent_idx, action))
        if self.catch_exceptions:
            try:
                next_state = self.game.state.generateSuccessor(self.agent_idx, action)
            except Exception as data:
                self.game.mute(self.agent_idx)
                self.game._agentCrash(self.agent_idx)
                self.game.unmute()
                return
        else:
            next_state = self.game.state.generateSuccessor(self.agent_idx, action)

        reward = next_state.getScore() - self.game.state.getScore()
        self.game.state = next_state

        # Update self.game.gameOver
        self.game.rules.process(self.game.state, self.game)
        
        done = self.game.gameOver # set by rules/game.state during rules.process
        info = {} if not done else {'game': self.game} # return game for recording
        
        # It's the next agent's move
        self.agent_idx = (self.agent_idx + 1) % self.num_agents

        return (next_state, reward, done, info)

    def render(self, mode='human'):
        '''
        '''
        if not self.display_initialized:
            self.display.initialize(self.game.state.data)
            self.display_initialized = True

        self.display.update(self.game.state.data)

    def getPossibleActions(self, state=None, agent_idx=None):
        '''
        state: defaults to current environment state.
        agent_idx: defaults to current environment actor.
        '''
        if agent_idx is None:
            idx = self.agent_idx
            
        if state is not None:
            return state.getLegalActions(agentIndex=idx)
        else:
            return self.game.state.getLegalActions(agentIndex=idx)
        
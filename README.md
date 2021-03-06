# gym-berkeley-reinforcement



## Introduction

This is a OpenAI Gym wrapper around Berkeley's CS188 Intro to AI Pacman Project 3: Reinforcement Learning.

Links to the Berkeley project:

- Course website: https://inst.eecs.berkeley.edu/~cs188/fa18/
- RL project page: https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html
- RL code download link: https://inst.eecs.berkeley.edu/~cs188/fa18/assets/files/reinforcement.zip


I thought it would be valuable to complete to agents from the course and then be able to reuse them with other Gym environments. 

Currently Gridworld is mostly done. Pacman is somewhat done and Crawler are pending.


## To Do

1. Port the timeout and exception handling from `Game.run()` to `pacmanapp.run_game()`

## Installation

Clone this repository.

Install with pip from within the directory to install the packages.

For example: `pip install -e .`


## Usage

### Pacman

Play an interactive game of pacman:

```
python pacmanapp.py
```

Watch a greedy pacman agent face off against the ghosts
on the smallGrid layout, first playing 9 games quietly before displaying a game

```
python pacmanapp.py -l smallGrid -p GreedyAgent -n 10 -x 9
```

All the bells and whistles:

```
python pacmanapp.py -l smallClassic -p ApproximateQAgent  -a extractor=SimpleExtractor -z2 -t -q -g DirectionalGhost -n100 -x90 -k1 -f
```

Simple example of a game loop:

```python
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
```

### Gridworld

The usage of `gridworldapp.py` is meant to be similar or identical to the original `gridworld.py` CLI:

```
python gridworldapp.py --agent q --grid=CliffGrid --episodes=10 --epsilon=0.1 --discount=0.99 
```

A trivial example of making and using a gridworld environment:

```python
import gym
import gymberkeleyrl
env = gym.make("Gridworld-MazeGrid-v0")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        done = False
        observation = env.reset()
```

## Design Choices

An major design goal was to make minimal changes to the existing Berkeley code. As such
changes were only made to the existing code to make it a python 3 package.

The overall structure of application like `pacman.py` or `gridworld.py` can be logically 
divided into an agent, an environment, and a coordinator.
- The agent is responsible for taking actions given a state and learning from experience,
  if it learns. 
- The environment keeps track of the current state, responding to actions by returning the
  next state and reward.
- The coordinator is responsible for reading configuration, constructing agents and the
  environment, and running the game loop.

In the OpenAI Gym API, the environment is responsible for rendering (i.e. displaying) the 
environment on request, via `render`. Also there is no specific mechanism for querying the legal
actions for a given state. 


### Pacman

Pacman can be seen as a multi-agent game. The Github issue,
https://github.com/openai/gym/issues/934, has many useful ideas for implementing a multi-agent
Gym environment. 

To interact with classes like `Game` and `ClassicGameRules` which vary their behavior
based on the agent index, `PacmanEnv` tracks the index of the player for the current step just by
incrementing an index (modulo the number of players). The environment assumes that pacman take 
the first action, then the ghosts, then pacman, and so on.

Like Gridworld, `env.getPossibleActions` returns the legal actions for the current state and the current
agent index, unless a state or agent index is passed to the function.


### Gridworld

When an illegal action is sent to the environment in `gridworld.py`, an Exception is raised.
Therefore an agent must send a legal action, unlike some other RL environments where any action is legal
in any state. Two possible ways to communicate the possible actions to the agent are:

1. Have a separate `getPossibleActions` method which returns the possible actions for a state.
   All actions are contained in `env.action_space.objects`.
2. Update `env.action_space` to reflect the possible actions. How will a naive agent know all possible
   actions? Either it would have to track actions itself or the environment could have a method for that.

The first choice was taken, since it makes `actionFn` simple and compatible with `display.displayValues()`.


## Future Work

The current Gym environments for Pacman and Gridworld do not use the standard numerical Space classes
for `action_space` and `observation_space`. Gridworld uses and `ObjectSpace`, representing a set
of discrete non-numerical objects. Pacman ignored `action_space` and `observation_space` entirely.

To make these Gym environments more typical of other environments or 
friendlier to 3rd party deep RL agents which expect numerical spaces:

- Make the `action_space` a Discrete space. The orchestration code (e.g. `pacmanapp.py`) 
  can translate into the actions (e.g. 'north', 'south', etc.) the Berkeley agent code understands.
- Possibly allow agents to make illegal actions, to accomodate 3rd party agents and to deprecate the 
  `getPossibleActions` method.
- Make the `observation_space` a Box space representing the board as 2d matrix, where each
  position encodes what is present at that position on a board. The current pacman observation is 
  a complex GameState object which includes references to internal game objects. A translation layer
  from a Box to a GameState-like object could accomodate existing agent code.
  
Changes to the agents that would make them more compatible with other Gym environments:

- Change the Berkeley pacman agent code to
  - take an `actionFn` instead of getting the possible actions from the `state`.
  - remove the `observationFunction` which computes the reward from 
    the `state.getScore() - last_state.getScore()`
    and replace with the existing `observeTransition`, using the reward computed by
    the orchestration code. See example below.


## Alterations to Berkeley Code

To make the original code in the `reinforcement/` directory a package:
- an `__init__.py` file was added to `reinforcement/` 
- relative package imports, e.g. `from . import util`, were added to the modules.
- layouts were added as `package_data` to `setup.py`
- The `layout.py` module was changed to access the package data.
- loadAgent` was rewritten in `pacmanapp.py` to look in a defined list of agent modules instead of 
  searching the current dir for files matching `*Agent.py` and importing them. Perhaps there is a more
  dynamic way of searching the package for modules containing agent classes.

The `2to3` conversion script introduced a bug in `textGridworldDisplay` by converting `map(None, *foo)`
to `list(*foo)` instead of `list(zip(*foo))`.

The `2to3` script did not convert bare string exceptions like `raise 'OH NO!'` to regular exceptions like `raise Exception('OH NO!')`.


## Other Gym Environments

Berkeley RL Environments

https://github.com/rlworkgroup/garage/blob/master/garage/envs/grid_world_env.py


## Notes on Berkeley Code Flow and OpenAI Gym

Consider the games pacman and gridworld.

Both games have boards (in `layouts/` for pacman and in `gridworld.py` for gridworld).

Both games have feature extractors, in `featureExtractors.py`, though gridworld did not use feature extractors until I added them for the last assignment.

There is a main script for each game, `pacman.py` and `gridworld.py`. 

The main script for `gridworld.py`:
- parses command line options, like the agent to use, the options for the agent (like number of steps), how long to train, what board to use, etc.
- imports agents implemented in `qlearningAgents.py`, `valueIterationAgents.py`, `reinforceAgents.py`, `sarsaNAgents.py`.
- constructs and configures an MDP for a grid specified on the command line
- constructs a display for the MDP
- construct an agent:
  - a dynamic programming agent is passed the MDP.
  - a learning agent is passed an 'actionFn' which returns the allowed actions for a state.
  - a function approximation agent is passed a feature extractor.
- construct decisionCallback using agents `getAction` method for agents that act. Callback is used to get an agent's action for a state during `runEpisode()`. 
- construct displayCallback, etc., also used during `runEpisode()`.
- run episodes using `runEpisode()`:
  - episode is started. reset environment. get initial state.
  - loop: 
    - display state.
    - pass state to agent and get action, via `decisionCallback`.
    - pass action to environment, get next state and reward.
    - pass (state, action, nextState, reward) to agent via `observeTransition` method, so it can learn.
- instantiates an agent based on options, including feature extractor.
- instantiates a game, passing it agent, display, board, etc. The game

The main script for `pacman.py`:

- read options
- construct game layout
- construct agent
- make ghost agents
- construct graphics: null, text, graphical
- `runGames()` is passed layout, agent, ghosts, display, numGames, etc.
  - construct ClassicGameRules
  - loop for numGames
    - use null display for training games, otherwise use display
    - construct game (see `game.py`)
    - `game.run()`
      - tell agents game started.
      - loop until game over
        - for each agent:
          - allow agent to generate an "observation" from game state. otherwise use game state as observation. `observationFunction` is where learning agents learn.
            - learningAgents.ReinforcementAgent class has pacman specific `observationFunction` that calculates reward and calls `observeTransition`.
          - give agent observation and get action
          - execute action and get next state
      - tell agents game ended.
      
The OpenAI Gym approach:

There is a (possibly parameterized) environment, env. That environment has:

- `reset()` initializes/resets the environment and returns the first observation
  - In gridworld, this is split into `environment.reset()` and `environment.getCurrentState()`
  - In pacman, in `pacman.runGames()` a new environment ('game') is created for each episode from the rules: `game = rules.newGame( layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)`
- `step(action)` advances the environment based on the action and returns the next observation, reward, done flag, and info. 
  - In gridworld, in the training loop, if `environment.getPossibleActions(state)` is empty, episode is done.
  - In gridworld, `environment.doAction(action)` returns next_state and reward.
  - In pacman, `self.state.generateSuccessor(agentIndex, action)` returns the next state.
  - In pacman, the reward is calculated by the agent in `learningAgents.ReinforcementAgent.observationFunction(state)` from the difference of `state.getScore` and `self.last_state.getScore()`. 
  - In pacman, the previous state is tracked by the agent.
  - In pacman, the done flag is in `Game.gameOver`.
- `render()` displays the environment in its current state
  - In pacman, `self.display.update( self.state.data )` is used by `game.run()` to display the current state.
  - In gridworld, `display(state)` is used in `runEpisode()` to display the game state.- `action_space`
- `env.action_space` describes all actions in the env, but not the actions valid for a state.
  - In gridworld, the valid actions for a state are `environment.getPossibleActions(state)`. No possible action means the episode is done.

Gridworld has an environment, `gridworld.GridworldEnvironment(mdp)`, constructed with the MDP. 
Pacman has an environment which is a `game.Game` and the `pacman.GameState` it contains. `pacman.ClassicGameRules` creates the game and game state.

Other things:

How does agent know legal actions? In gridworld and pacman, agent.getAction() calls self.getLegalActions() which calls self.actionFn(state)
- gridworld: `actionFn = lambda state: mdp.getPossibleActions(state)`. The agent is initialized with an actionFn that gets the legal actions from the MDP.
- pacman: in `learningAgents.ReinforcementAgent.__init__`, `if actionFn == None: actionFn = lambda state: state.getLegalActions()`. The agent is not initialized with an actionFn, so by default it gets the legal actions from the state. The GameState uses `packman.PacmanRules.getLegalActions(state)`.

Episode loops:
- in gridworld, `gridworld.runEpisode` runs the episode loop, starting the environment and agent, getting state from the environment, displaying the state, getting action from the agent, getting next state and reward from the environment, passing (state, action, nextState, reward) to agent, and eventually stopping the agent.
- `Game.run()` runs the episode/training loop, which gets the state, passes it to the agent to get the action, passes the state to display to render the game, passes the action to the It gets the state from `self.state`. 
- In pacman, the game loop runs over the agent and the ghost agents, incrementing the state every time. Therefore the reward for the agent is the score of the current state minus the score from n states previous, where n is the total number of agents (num ghosts + 1).

Big picture:

- there is an agent
  - return action via getAction(state), which knows about legal actions by querying environment
  - learn params via observeTransition(state, action, nextState, reward)
  - maybe initialize and finish episodes.
- there is an environment
  - transitions to next state given action
  - tracks if episode is done. 
  - tracks reward, sort of.
  - can reset (possibly)
  - knows about legal actions for a state
- there is a training/episode loop that coordinates between display, agent and environment
  - update display
  - get action
  - update environment
  - update agent

What it would take to make gridworld an openai environment:
- updates to environment
  - Move display into environment, add `render()`.
  - explicitly return done flag from environment
  - Change `environment.reset()` to return `environment.getCurrentState()`
- updates to agent
  - how would agent know legal moves?
- updates to episode loop
  - call env.render()
  - not much else.

What it would take to make pacman an openai environment:
- updates to environment
  - multi-agent gym ideas: https://github.com/openai/gym/issues/934
  - Move display into environment, add `render()`.
  - explicitly return done flag from environment instead of using `game.gameOver`.
  - explicitly return reward from env, instead of agent using (`state.getScore() - last_state.getScore()`).
    - environment returns `reward = next_state.getScore() - state.getScore()`.
      The episode loop sum up the rewards
      across the steps of the agent and ghosts, so agent gets the same reward. I.e. loop
      tracks the reward (state.getScore - last_state.getScore) instead of agent or environment. 
  - Change `environment.reset()` to use rules to create new game and game state.
  - Add rules, game, and game state to the environment, so it can "reset", etc.
  - Update the environment to handle illegal moves
  - Update the environment to say legal moves
    - this can be accomplished by changing `env.action_space` in `step()`.
- updates to agent
  - how would agent know legal moves?
- updates to episode loop
  - call env.render()
  - remove episode loop from Game class.




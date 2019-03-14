# gym-berkeley-reinforcement

## To Do

- copy value agent, qlearning agent into repo so I can see they work.


## Introduction

This is a OpenAI Gym wrapper around Berkeley AI Pacman Project 3: Reinforcement

See http://inst.eecs.berkeley.edu/~cs188/pacman/reinforcement.html for the original project.


## Installation

Clone this repository.

Download and unzip http://ai.berkeley.edu/projects/release/reinforcement/v1/001/reinforcement.zip.

Place the directory `reinforcement` inside the `gym-pacman/gym_pacman` directory.


## Usage


## Other Gym Environments

Berkeley RL Environments

https://github.com/rlworkgroup/garage/blob/master/garage/envs/grid_world_env.py


## Thoughts on Berkeley Reinforcement Code Flow and OpenAI Gym

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




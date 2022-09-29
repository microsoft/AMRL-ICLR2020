'''
An Env designed to test different aspects of long term memory.
Used in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete, Box, Tuple
from gym.envs.registration import EnvSpec
from gym.utils import seeding

# Ray imports. (If you want to use ray, you can do this by removing where these are used):
# Used to support a multi-agent rllib env:
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# Used for getting global var info (e.g. number of steps trained) from rllib. Only needed if task_switch_after in config is not None:
from ray.rllib.env.env_context import EnvContext

import os
import yaml
import numpy as np
from time import time
from shutil import copy2
from pathlib import Path
from collections import defaultdict

TMAZE_ENV_KEY = "tmaze-v0"

class Actions():
    """Enum class to define actions"""
    LEFT, RIGHT, UP, DOWN = range(4) # Must be permutations of [0...num_actions-1]
    CHECK_UP, CHECK_DOWN = range(2) # Must be permustations of [0,1]

class Indicator_Dir():
    """Enum class to define the values (directions) of the longterm indicator"""
    UP, DOWN = Actions.UP, Actions.DOWN # For making equality convenient

class Check_Dir():
    """Enum class to define the values (directions) of the intermediate indicators"""
    UP, DOWN = Actions.CHECK_UP, Actions.CHECK_DOWN # For making equality convenient

class Obs():
    def __init__(self, check_up, check_down):
        """Class to define observations"""
        self.START         = [1,0,0]
        self.START_UP      = [1,1,0]
        self.START_DOWN    = [1,-1,0]
        self.END           = [0,0,1]
        self.END_UP        = [0,1,1]
        self.END_DOWN      = [0,-1,1]
        self.MIDDLE        = [0,0,0]
        self.MIDDLE_UP     = [0,1,0]
        self.MIDDLE_DOWN   = [0,-1,0]
        # The following indicators are appeneded to above if self.intermediate_indicators:
        self.CHECK_UP   = check_up
        self.CHECK_DOWN = check_down
        self.END_CHECK  = 0 # Sepcial case for the end, if there is not be an indicator there
        self.CHECK2OBS  = {
            Check_Dir.UP: self.CHECK_UP, 
            Check_Dir.DOWN: self.CHECK_DOWN,
        }
        self.OBS2CHECK = { # Useful for testing
            self.CHECK_UP: Check_Dir.UP,
            self.CHECK_DOWN: Check_Dir.DOWN,
        }

        self.OBS_LIST = [self.START, self.START_UP, self.START_DOWN, self.END, self.END_UP, self.END_DOWN, 
                    self.MIDDLE, self.MIDDLE_UP, self.MIDDLE_DOWN]
        self.CHECK_LIST = [[self.CHECK_UP], [self.CHECK_DOWN], [self.END_CHECK]]

    def get_observation_space(self, intermediate_indicators, pos_enc, wave_encoding_len, timeout, num_indicators_components=None):
        if intermediate_indicators:
            assert num_indicators_components is not None, num_indicators_components
            main_obs_low = np.array(self.OBS_LIST).min(axis=0)
            main_obs_hi = np.array(self.OBS_LIST).max(axis=0)
            indicator_component_low = np.array(self.CHECK_LIST).min(axis=0)
            indicator_component_hi = np.array(self.CHECK_LIST).max(axis=0)
            indicator_low = np.broadcast_to(indicator_component_low, (num_indicators_components))
            indicator_hi = np.broadcast_to(indicator_component_hi, (num_indicators_components))
            low = np.concatenate((main_obs_low, indicator_low), axis=-1)
            hi = np.concatenate((main_obs_hi, indicator_hi), axis=-1)
        else:
            low = np.array(self.OBS_LIST).min(axis=0)
            hi = np.array(self.OBS_LIST).max(axis=0)

        if pos_enc:
            if wave_encoding_len is None:
                enc_low = np.array([0])
                enc_hi = np.array([timeout])
            else:
                enc_low = np.array([-1 for _ in range(wave_encoding_len)])
                enc_hi = np.array([1 for _ in range(wave_encoding_len)])
            low = np.concatenate((low, enc_low), axis=-1)
            hi = np.concatenate((hi, enc_hi), axis=-1)

        return Box(low, hi, dtype=np.float32)

class TMaze(gym.Env):
    """
    T-maze in which an indicator along a corridor corresponds to the goal location
    at the end of the corridor.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:
            allow_left (bool): Whether the agent should be able to step left (backwards)
            timeout (int): How many steps the agent is allowed to take before the episode terminates
            timeout_reward (float): The reward the agent receives form a timeout
            maze_length (int): The length of the maze (number of steps to solve optimally)
            indicator_pos (int): The location [0,maze_length) to place the indicator meant to be remembered
            success_reward (float): The reward for choosing the correct action (up or down) at the end of the maze
            fail_reward (float): The reward for choosing the incorrect action (up or down) at the end of the maze
            check_reward (float): If there are intermediate checks (tasks), the reward per step for getting it correct
            persistent_reward (float): A reward given per timestep
            force_final_decision (bool): Whether the agent must choose up or down at the end of the maze
            force_right (bool): Whether the agent goes to the right in the corridor (progreses) regardless of action
            intermediate_checks (bool): Whether there is an additional task of reproducing an observation at each step
            intermediate_indicators(bool): Whether there are additional observations (bitsup or down) at each step.
            reset_intermediate_indicators (bool): Whether the intermediate observation reset between episodes
            final_intermediate_indicator (bool): Whether there should be an intermediate observation at the end.
            task_switch_after (int): Step number to trigger a predfined change in config options
            per_step_reset (bool): Whether the intermediate obseravtoins reset between steps
            num_indicators_components (int): Number of dimensions in the intermediate observations used for checks
            frac_correct_components_for_check (int): The number of dimensions in the intermediate observaiton the agent
                                                     must get correct to be progressed to the next position
            reward_per_correct_component (bool): Whether the agent receives reward per correct reconstruction of the
                                                 intermediate observation, or only if it gets it correct and progressed
            flipped_indicator_pos (int): If specified, an indicator will be placed here repreenting the opposite
                                         direction as that of the standard long term indicator
            wave_encoding_len (int or None): If specified and pos_enc is True, then the timestep, encoded using
                                             sine and cosine waves with this number of dimensions will be added to 
                                             the observation
            pos_enc (bool): Whether to outpute the timestep as part of the observation
            correlated_indicator_pos (int or None): If specified, another long-term indicator will be placed here,
                                                    and the agent must go up at the end iff the first indicator
                                                    and this indicator together occur with the pattern: UP, DOWN
            check_up (int): An int used to encode an intermedaite check "up" observation (per dimension)
            check_down (int): An int used to encode an intermedaite check "down" observation (per dimension)
            maze_length_upper_bound (int or None): If specified, a random maze length will be sampled
                                                   uniformly at random between maze_length and maze_length_upper_bound,
                                                   to test generalization
    """

    def __init__(self, config):
        required_args = set(["allow_left",
                              "timeout",
                              "timeout_reward",
                              "maze_length",
                              "indicator_pos",
                              "success_reward",
                              "fail_reward",
                              "check_reward",
                              "persistent_reward",
                              "force_final_decision",
                              "force_right",
                              "intermediate_checks",
                              "intermediate_indicators",
                              "reset_intermediate_indicators",
                              "final_intermediate_indicator",
                              "task_switch_after",
                              "per_step_reset",
                              "num_indicators_components",
                              "frac_correct_components_for_check",
                              "reward_per_correct_component",
                              "flipped_indicator_pos",
                              "wave_encoding_len",
                              "pos_enc",
                              "correlated_indicator_pos",
                              "check_up",
                              "check_down",
                              "maze_length_upper_bound",])
        given_args = set(config.keys())
        self.OBS = Obs(check_up=config["check_up"], check_down=config["check_down"])
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)
        self.task_switch_after = config["task_switch_after"]
        self.force_final_decision = config["force_final_decision"]
        self.force_right = config["force_right"]
        self.allow_left = config["allow_left"]
        self.intermediate_checks = config["intermediate_checks"]
        self.intermediate_indicators = config["intermediate_indicators"]
        self.reset_intermediate_indicators = config["reset_intermediate_indicators"]
        self.final_intermediate_indicator = config["final_intermediate_indicator"]
        self.per_step_reset = config["per_step_reset"]
        self.num_indicators_components = config["num_indicators_components"]
        self.frac_correct_components_for_check = config["frac_correct_components_for_check"]
        self.reward_per_correct_component = config["reward_per_correct_component"]
        self.wave_encoding_len = config["wave_encoding_len"]
        self.pos_enc = config["pos_enc"]
        assert not (self.force_right and self.allow_left), "Cannot force right action and allow left action"
        assert not (self.force_right and self.intermediate_checks), "Cannot force right action and do intermediate checks"
        assert not (self.intermediate_checks and not self.intermediate_indicators), "Intermediate indicators required to do intermediate checks"
        self.timeout = config["timeout"]
        self.timeout_reward = config["timeout_reward"]
        self.success_reward = config["success_reward"]
        self.fail_reward = config["fail_reward"]
        self.check_reward = config["check_reward"]
        self.persistent_reward = config["persistent_reward"]
        self.maze_len = config["maze_length"]
        assert self.maze_len >= 1, self.maze_len
        self.maze_length_upper_bound = config["maze_length_upper_bound"]
        if self.maze_length_upper_bound is not None:
            self.maze_length_lower_bound = self.maze_len
            assert self.maze_length_upper_bound >= self.maze_len, (self.maze_length_upper_bound, self.maze_len)
        self.indicator_pos = config["indicator_pos"]
        assert self.indicator_pos >= 0 and self.indicator_pos <= self.maze_len-1, self.indicator_pos
        
        self.flipped_indicator_pos = config["flipped_indicator_pos"]
        self.correlated_indicator_pos = config["correlated_indicator_pos"]
        indicator_at_endpoint = (self.indicator_pos == 0) or (self.indicator_pos == (self.maze_len-1))
        if self.flipped_indicator_pos is not None:
            assert self.correlated_indicator_pos is None, "Cannot have correlated indicator and flipped indicator currently"
            flipped_indicator_at_endpoint = (self.flipped_indicator_pos == 0) or (self.flipped_indicator_pos == (self.maze_len-1))
            assert not (indicator_at_endpoint or flipped_indicator_at_endpoint), \
                   "This is probably won't test the order dependence I want, so not implemented."
            assert not (self.flipped_indicator_pos == self.indicator_pos), "Cannot have both long term indicators in the same location"
        if self.correlated_indicator_pos is not None:
            correlated_indicator_at_endpoint = (self.correlated_indicator_pos == 0) or (self.correlated_indicator_pos == (self.maze_len-1))
            assert not (indicator_at_endpoint or correlated_indicator_at_endpoint), \
                   "This is probably won't test the order dependence I want, so not implemented."
            assert not (self.correlated_indicator_pos == self.indicator_pos), "Cannot have both long term indicators in the same location"

        if self.task_switch_after is not None:
            # Override settings
            self.intermediate_checks = True
            self.intermediate_indicators = True
            self.reset_intermediate_indicators = True
            self.per_step_reset = True
            self.final_intermediate_indicator = False
            self.force_right = False
            self.allow_left = False
            self.force_final_decision = False
            self.timeout = self.maze_len - 1
            self.success_reward = 0.0
            self.fail_reward = 0.0

        indicator_actions = []
        if self.intermediate_checks:
            indicator_actions = [Discrete(2) for _ in range(self.num_indicators_components)]
        self.action_space = Tuple([Discrete(4)] + indicator_actions) # Directions in maze + indicator checks
        self.observation_space = self.OBS.get_observation_space(self.intermediate_indicators, self.pos_enc, self.wave_encoding_len, self.timeout,
                                                           num_indicators_components=self.num_indicators_components)
        self._spec = EnvSpec(TMAZE_ENV_KEY)

        if type(config) is EnvContext:
            self.global_vars = config.global_vars
        else:
            self.global_vars = {"timestep" : 0} # Dummy value for testing

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_random_indicator(self):
        return self.np_random.choice([Indicator_Dir.UP, Indicator_Dir.DOWN], p=[0.5, 0.5])

    def reset(self):
        if self.task_switch_after is not None \
        and "timestep" in self.global_vars \
        and self.global_vars["timestep"] >= self.task_switch_after:
            # Override settings
            self.intermediate_checks = False
            self.check_reward = 0.0
            self.intermediate_indicators = True # Leave indicators visible, but no checks
            self.reset_intermediate_indicators = False
            self.final_intermediate_indicator = False
            self.force_right = True
            self.allow_left = False
            self.force_final_decision = False
            self.timeout = 150
            self.success_reward = 4.0
            self.fail_reward = -0.1

        if self.maze_length_upper_bound is not None:
            self.maze_len = self.np_random.randint(low=self.maze_length_lower_bound, high=self.maze_length_upper_bound+1)
        self.timestep = 0
        self.cur_pos = 0
        self.indicator = self.get_random_indicator()
        
        self.secondary_indicator_pos = None
        if self.flipped_indicator_pos is not None:
            self.secondary_indicator_pos = self.flipped_indicator_pos
            self.secondary_indicator = Indicator_Dir.UP if self.indicator == Indicator_Dir.DOWN else Indicator_Dir.DOWN
        elif self.correlated_indicator_pos is not None:
            self.secondary_indicator_pos = self.correlated_indicator_pos
            self.secondary_indicator = self.get_random_indicator()  # This is correlated with goal in that DD,UU,DU -> D; UD -> U

        if self.intermediate_indicators:
            random = self.np_random if self.reset_intermediate_indicators else np.random.RandomState(0)
            self.checks = random.choice([Check_Dir.UP, Check_Dir.DOWN], 
                                         size=(self.maze_len, self.num_indicators_components), p=[0.5, 0.5])
            if not self.final_intermediate_indicator:
                self.checks = list(self.checks)
                self.checks[-1] = None
        return self.get_obs()

    def add_check(self, obs):
        """ Adds the check to the end of obs if self.intermediate_indicators """
        
        if not self.intermediate_indicators:
            return obs

        if self.get_cur_check() is None:
            assert self.cur_pos == self.maze_len-1, self.cur_pos
            check_obs = [self.OBS.END_CHECK for _ in range(self.num_indicators_components)]
        else:
            check_obs = [self.OBS.CHECK2OBS[c] for c in self.get_cur_check()]

        return obs + check_obs

    def pos_wave_encoding(self, p, l, c=10000):
        """ Calculates a positional encoding of length l for position p inline with https://arxiv.org/pdf/1706.03762 """
        enc = [None for _ in range(l)]
        for i in range(l):
            v = p / (c**(2*i / l))
            enc[i] = np.sin(v) if (i%2) == 0 else np.cos(v)
        return enc

    def add_positional_encoding(self, obs):
        """ Adds a positional encoding to the input """
        if not self.pos_enc:
            return obs
        return obs + (self.pos_wave_encoding(self.timestep, self.wave_encoding_len, self.timeout) if (self.wave_encoding_len is not None) \
               else [self.timestep])

    def get_obs_without_check(self):
        """Returns the correct current observation assuming not self.intermediate_indicators."""
        
        if self.cur_pos == 0:
            # Start
            if self.indicator_pos != self.cur_pos: # Indicator not here
                return self.OBS.START
            if self.indicator == Indicator_Dir.UP: # Indicator up
                return self.OBS.START_UP
            assert self.indicator == Indicator_Dir.DOWN, self.indicator
            return self.OBS.START_DOWN # Indicator down

        if self.cur_pos == self.maze_len-1:
            # End
            if self.indicator_pos != self.cur_pos: # Indicator not here
                return self.OBS.END
            if self.indicator == Indicator_Dir.UP: # Indicator up
                return self.OBS.END_UP
            assert self.indicator == Indicator_Dir.DOWN, self.indicator
            return self.OBS.END_DOWN # Indicator down

        assert self.cur_pos > 0 and self.cur_pos < self.maze_len-1, self.cur_pos
        # Middle
        if (self.indicator_pos != self.cur_pos) and (self.secondary_indicator_pos != self.cur_pos): # Indicator not here
            return self.OBS.MIDDLE # Regular middle (no indicator)
        if self.indicator_pos == self.cur_pos:
            ind = self.indicator
        elif self.secondary_indicator_pos == self.cur_pos:
            ind = self.secondary_indicator
        if ind == Indicator_Dir.UP: # Indicator up
            return self.OBS.MIDDLE_UP
        assert ind == Indicator_Dir.DOWN, ind
        return self.OBS.MIDDLE_DOWN # Indicator down  

    def get_obs(self):
        """Returns the correct current observation."""
        return self.add_positional_encoding(self.add_check(self.get_obs_without_check()))

    def get_cur_check(self):
        return self.checks[self.cur_pos]

    def move_right(self):
        self.cur_pos += 1
        self.cur_pos = min(self.cur_pos, self.maze_len-1)

    def move_left(self):
        self.cur_pos -= 1
        self.cur_pos = max(self.cur_pos, 0)

    def reset_cur_check(self):
        if self.checks[self.cur_pos] is not None:
            self.checks[self.cur_pos] = self.np_random.choice([Check_Dir.UP, Check_Dir.DOWN], 
                                                              size=(self.num_indicators_components), p=[0.5, 0.5])

    def step(self, action):
        assert len(action) == (1+self.num_indicators_components if (self.intermediate_checks or self.task_switch_after is not None) else 1), action
        a = action[0] # The main (directional) action

        # timeout if necessary
        if self.timeout is not None and self.timestep > self.timeout-1:
            if self.reset_intermediate_indicators and self.per_step_reset:
                self.reset_cur_check()
            return self.get_obs(), self.timeout_reward, True, {}

        # options that force decions (by changing action taken)
        if self.force_final_decision and (self.cur_pos == self.maze_len-1):
            if a == Actions.LEFT:
                a = Actions.UP
            elif a == Actions.RIGHT:
                a = Actions.DOWN
        if self.force_right and (self.cur_pos < self.maze_len-1):
            a = Actions.RIGHT

        # check the direction of a and respond appropriately
        done = False
        reward = self.persistent_reward
        moved_left = False
        if a == Actions.LEFT:
            if self.allow_left:
                self.move_left()
                moved_left = True
        elif a == Actions.RIGHT:
            if not self.intermediate_checks:
                self.move_right()
        else:
            assert a in [Actions.UP, Actions.DOWN]
            if self.cur_pos == self.maze_len-1: # At end
                # Tell whether successful
                if self.correlated_indicator_pos is None:
                    success = (a == self.indicator)# Only one indicator and agent is correct
                else:
                    indicator_pattern = (self.indicator, self.secondary_indicator) # DD,UU,DU -> D; UD -> U
                    up_patterns = [(Actions.UP, Actions.DOWN)]
                    down_patterns = [(Actions.DOWN, Actions.DOWN), (Actions.UP, Actions.UP), (Actions.DOWN, Actions.UP)]
                    assert indicator_pattern in (up_patterns + down_patterns), indicator_pattern
                    success = ((a == Actions.UP) and (indicator_pattern in up_patterns)) \
                          or ((a == Actions.DOWN) and (indicator_pattern in down_patterns))
                # Give reward and addign done
                reward += self.success_reward if success else self.fail_reward
                done = True
        
        # Deal with intermediate checks
        # Only do this in the case that you havent already moved left (which takes presedence) and not at end
        # (If you could get intermediate reward at end, you might prefer to not end epsode.)
        if self.intermediate_checks and (self.cur_pos < self.maze_len-1) and not moved_left:
            indicator_actions = action[1:]
            num_correct = np.sum(self.get_cur_check() == indicator_actions)
            num_correct_needed = np.ceil(self.frac_correct_components_for_check * self.num_indicators_components)
            if num_correct >= num_correct_needed:
                check_r = self.check_reward
                if self.reward_per_correct_component:
                    check_r *= (num_correct/self.num_indicators_components)
                reward += check_r
                self.move_right()

        # Update timestep (and intermediate indicator) and return
        self.timestep += 1
        if self.intermediate_indicators and self.reset_intermediate_indicators and self.per_step_reset:
            self.reset_cur_check()
        return self.get_obs(), reward, done, {}

    def render(self, mode='human'):
        lines = [['x' for _ in range(self.maze_len+2)] for _ in range(5)]
        lines[2][1:-1] = [' ' for _ in range(self.maze_len)]
        lines[2][self.indicator_pos+1] = 'I'

        if self.indicator == Indicator_Dir.UP:
            lines[1][-2] = 'g'
            lines[3][-2] = ' '
        else:
            lines[1][-2] = ' '
            lines[3][-2] = 'g'

        lines[2][self.cur_pos+1] = 'a'

        print()
        print('\n'.join([''.join(line) for line in lines]))  


class MultiEnvTMaze(MultiAgentEnv):
    # Agents are in their own mazes. They do not affect eachother. Can still be used to test joint Q value.
    # Note, agent IDs are given as ints, not string, as in some other examples.
    def __init__(self, env_config, num=1):
        self.agents = [TMaze(env_config) for _ in range(num)]
        self.dones = set()
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space

    def reset(self):
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            # Freeze sub-env if it is done.
            if i in self.dones:
                obs[i], rew[i], done[i], info[i] = self.agents[i].get_obs(), 0.0, True, {}
            else:
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

def grouped_env_creator(num_agents, env_config):
    sample_tmaze = TMaze(env_config)
    groups = {"group_1": [a for a in range(num_agents)]} # Note, using agent ids as ints in MultiEnvTMaze
    obs_space = Tuple([sample_tmaze.observation_space for a in range(num_agents)])
    act_space = Tuple([sample_tmaze.action_space for a in range(num_agents)])
    multi_agent_env_creator = lambda ignore_config: MultiEnvTMaze(env_config, num=num_agents).with_agent_groups(
        groups, obs_space=obs_space, act_space=act_space)
    return multi_agent_env_creator

'''
Test cases for t_maze env
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from numpy.testing import assert_allclose

from ray.rllib.examples.t_maze import TMaze, MultiEnvTMaze, grouped_env_creator, Actions, Indicator_Dir, Check_Dir, Obs

def correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, num_components, 
                                       final_intermediate_indicator=False, at_end=False):
    indicator_obs = None
    if (not final_intermediate_indicator) and at_end:
        indicator_obs = [maze.OBS.END_CHECK for _ in range(num_components)]
    else:
        indicator_obs = [maze.OBS.CHECK_UP if c == Check_Dir.UP else maze.OBS.CHECK_DOWN for c in cur_check]
    return correct_obs_no_check+indicator_obs

def a_2_tuple_a(maze, a, num_components, obs=None, wrong=False):
    """ Get corect tuple action from dir action plus indicator observation. 
        Take wrong indicator actions if wrong.
        Dummy indicator actions if obs is None. """
    DUMMY_CHECK_ACTION = Actions.CHECK_DOWN
    if obs is None:
        indicator_action = tuple(DUMMY_CHECK_ACTION for _ in range(num_components))
    else:
        i_obs = obs[-num_components:]
        if wrong:
            indicator_action = tuple(Actions.CHECK_DOWN if check_obs == maze.OBS.CHECK_UP else Actions.CHECK_UP \
                                     for check_obs in i_obs)
        else:
            d = maze.OBS.OBS2CHECK.copy()
            d[maze.OBS.END_CHECK] = DUMMY_CHECK_ACTION # Dummy action if end check (at end and no check to perform)
            indicator_action = tuple(d[check_obs] for check_obs in i_obs)

    return (a,) + indicator_action

class TMazeTest(unittest.TestCase):
    DEFAULT_CONFIG = {
        "force_final_decision" : False,
        "force_right" : False,
        "intermediate_checks" : False,
        "allow_left": True,
        "timeout": 1000,
        "timeout_reward": 0.0,
        "maze_length": 100,
        "indicator_pos": 0,
        "success_reward":4.0,
        "fail_reward": 0.0,
        "persistent_reward": 0.0,
        "check_reward": 0.1,
        "intermediate_indicators": False,
        "reset_intermediate_indicators": True,
        "final_intermediate_indicator": True,
        "task_switch_after": None,
        "per_step_reset": False,
        "num_indicators_components": 1,
        "frac_correct_components_for_check": 0.5,
        "reward_per_correct_component": False,
        "flipped_indicator_pos": None,
        "wave_encoding_len": None,
        "pos_enc": False,
        "correlated_indicator_pos": None,
        "check_up": 8,
        "check_down": 6,
        "maze_length_upper_bound": None
    }

    def testMultiAgentWrapper(self):
        OBS = Obs(check_up=TMazeTest.DEFAULT_CONFIG["check_up"], check_down=TMazeTest.DEFAULT_CONFIG["check_down"])
        for num_agents in range(1,5):
            # Create env
            multi_agent_env = grouped_env_creator(num_agents, TMazeTest.DEFAULT_CONFIG)(None)
            self.assertTrue(type(multi_agent_env.env.agents[0]) is TMaze)
            reset_joint_obs = multi_agent_env.reset()
            self.assertEqual(len(reset_joint_obs), 1)
            self.assertTrue("group_1" in reset_joint_obs)
            self.assertEqual(len(reset_joint_obs["group_1"]), num_agents)
            
            # Step to right
            right_action = {"group_1": [(Actions.RIGHT,) for a in range(num_agents)]}
            joint_obs, joint_r, joint_done, _ = multi_agent_env.step(right_action)
            for indivisual_obs in joint_obs["group_1"]:
                self.assertEqual(indivisual_obs, OBS.MIDDLE)
            self.assertEqual(joint_r["group_1"], 0)
            self.assertFalse(joint_done["group_1"])
            self.assertFalse(joint_done["__all__"])

            # Step first agent left and other agents agent right (if exists)
            joint_action = {"group_1": [(Actions.RIGHT,) for a in range(num_agents)]}
            joint_action["group_1"][0] = (Actions.LEFT,)
            joint_obs, joint_r, joint_done, _ = multi_agent_env.step(joint_action)
            for a, indivisual_obs in enumerate(joint_obs["group_1"]):
                if a == 0:
                    self.assertTrue(indivisual_obs in [OBS.START_UP, OBS.START_DOWN])
                else:
                    self.assertEqual(indivisual_obs, OBS.MIDDLE)
            self.assertEqual(joint_r["group_1"], 0)
            self.assertFalse(joint_done["group_1"])
            self.assertFalse(joint_done["__all__"])

            # Reset
            joint_obs = multi_agent_env.reset()
            for indivisual_obs in joint_obs["group_1"]:
                self.assertTrue(indivisual_obs in [OBS.START_UP, OBS.START_DOWN])

            # Walk to end
            for _ in range(200):
                joint_obs, r, joint_done, _ = multi_agent_env.step(right_action)
            for indivisual_obs in joint_obs["group_1"]:
                self.assertTrue(indivisual_obs, OBS.END)
            self.assertEqual(joint_r["group_1"], 0)
            self.assertFalse(joint_done["group_1"])
            self.assertFalse(joint_done["__all__"])

            # Step all agents up and make sure reward is correct
            joint_action = {"group_1": [(Actions.UP,) for a in range(num_agents)]}
            joint_obs, joint_r, joint_done, _ = multi_agent_env.step(joint_action)
            for indivisual_obs in joint_obs["group_1"]:
                self.assertTrue(indivisual_obs, OBS.END)
            reward = 0
            for agent_env in multi_agent_env.env.agents:
                reward += 4.0 if agent_env.indicator == Indicator_Dir.UP else 0.0
            self.assertEqual(joint_r["group_1"], reward)
            self.assertTrue(joint_done["group_1"])
            self.assertTrue(joint_done["__all__"])

            # Reset, walk to end, step up one at a time and make sure reward is correct
            multi_agent_env.reset()
            for _ in range(200):
                joint_obs, joint_r, joint_done, _ = multi_agent_env.step(right_action)
                self.assertFalse(joint_done["group_1"])
                self.assertFalse(joint_done["__all__"])
            for agent in range(num_agents):
                joint_action = {"group_1": [(Actions.RIGHT,) for a in range(num_agents)]}
                joint_action["group_1"][agent] = (Actions.UP,)
                self.assertFalse(joint_done["group_1"])
                self.assertFalse(joint_done["__all__"])
                joint_obs, joint_r, joint_done, _ = multi_agent_env.step(joint_action)
                for indivisual_obs in joint_obs["group_1"]:
                    self.assertTrue(indivisual_obs, OBS.END)
                proper_reward = 4.0 if multi_agent_env.env.agents[agent].indicator == Indicator_Dir.UP else 0.0
                self.assertEqual(joint_r["group_1"], proper_reward)
            self.assertTrue(joint_done["group_1"])
            self.assertTrue(joint_done["__all__"])

            # Rest, walk to end, step up one agent at a time. Make sure previous agents are frozen.
            # TODO: factor out code reuse between below and above
            multi_agent_env.reset()
            for _ in range(200):
                joint_obs, joint_r, joint_done, _ = multi_agent_env.step(right_action)
                self.assertFalse(joint_done["group_1"])
                self.assertFalse(joint_done["__all__"])
            for agent in range(num_agents):
                # Set action for previous agents to up as well
                joint_action = {"group_1": [(Actions.RIGHT,) if a > agent else (Actions.UP,) for a in range(num_agents)]}
                joint_action["group_1"][agent] = (Actions.UP,)
                self.assertFalse(joint_done["group_1"])
                self.assertFalse(joint_done["__all__"])
                joint_obs, joint_r, joint_done, _ = multi_agent_env.step(joint_action)
                for indivisual_obs in joint_obs["group_1"]:
                    self.assertTrue(indivisual_obs, OBS.END)
                proper_reward = 4.0 if multi_agent_env.env.agents[agent].indicator == Indicator_Dir.UP else 0.0
                self.assertEqual(joint_r["group_1"], proper_reward)
            self.assertTrue(joint_done["group_1"])
            self.assertTrue(joint_done["__all__"])

    def testMazeLen1(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["maze_length"] = 1
        maze = TMaze(new_conf)
        # Try left and right
        for action in [Actions.LEFT, Actions.RIGHT]:
            obs, r, done, _ = maze.step((action,))
            correct_obs = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.START_DOWN 
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
        # Reset
        obs = maze.reset()
        self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
        # Try up
        obs, r, done, _ = maze.step((Actions.UP,))
        reward = 4.0 if maze.indicator == Indicator_Dir.UP else 0.0
        self.assertEqual(reward, r)
        self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
        self.assertTrue(done)
        # Reset
        obs = maze.reset()
        self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
        # Try down
        obs, r, done, _ = maze.step((Actions.DOWN,))
        reward = 4.0 if maze.indicator == Indicator_Dir.DOWN else 0.0
        self.assertEqual(reward, r)
        self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
        self.assertTrue(done)

    def testRandLen(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["maze_length"] = low = 10
        new_conf["maze_length_upper_bound"] = hi = 15
        maze = TMaze(new_conf)
        lens = []
        for _ in range(20):
            obs = maze.reset()
            self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
            for i in range(maze.maze_len-2):
                obs, r, done, _ = maze.step((Actions.RIGHT,))
                self.assertEqual(0, r)
                self.assertTrue(obs == maze.OBS.MIDDLE)
                self.assertFalse(done)
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            self.assertEqual(0, r)
            self.assertTrue(obs == maze.OBS.END)
            self.assertFalse(done)
            obs, r, done, _ = maze.step((Actions.UP,))
            reward = 4.0 if maze.indicator == Indicator_Dir.UP else 0.0
            self.assertEqual(reward, r)
            self.assertTrue(obs == maze.OBS.END)
            self.assertTrue(done)
            lens.append(maze.maze_len)
            self.assertTrue(maze.maze_len <= hi and maze.maze_len >= low)
        self.assertTrue(not all([l == lens[0] for l in lens]))


    def testNoise(self):
        # Test intermediate indicators without checks
        for final_intermediate_indicator in [True, False]:
            new_conf = dict(TMazeTest.DEFAULT_CONFIG)
            new_conf["intermediate_checks"] = False
            new_conf["intermediate_indicators"] = True
            new_conf["reset_intermediate_indicators"] = True
            new_conf["num_indicators_components"] = n = 10
            new_conf["final_intermediate_indicator"] = final_intermediate_indicator
            maze = TMaze(new_conf)
            # Try left, up, down at start
            for action in [Actions.LEFT, Actions.UP, Actions.DOWN]:
                obs, r, done, _ = maze.step((action,))
                correct_obs_no_check = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                                       maze.OBS.START_DOWN 
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                                 final_intermediate_indicator=final_intermediate_indicator)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            # Walk to right 1 step
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            correct_obs_no_check = maze.OBS.MIDDLE
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                            final_intermediate_indicator=final_intermediate_indicator)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertEqual(maze.cur_pos, 1)
            self.assertFalse(done)
            # Try up and down and make sure you do NOT advance
            for action in [Actions.UP, Actions.DOWN]:
                obs, r, done, _ = maze.step((action,))
                correct_obs_no_check = maze.OBS.MIDDLE
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                                 final_intermediate_indicator=final_intermediate_indicator)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertEqual(maze.cur_pos, 1)
                self.assertFalse(done)
            # Walk to second to last step by going right
            for _ in range(97):
                obs, r, done, _ = maze.step((Actions.RIGHT,))
                correct_obs_no_check = maze.OBS.MIDDLE
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                                 final_intermediate_indicator=final_intermediate_indicator)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            # Step to end
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            correct_obs_no_check = maze.OBS.END
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=final_intermediate_indicator, at_end=True)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # Take correct actions
            correct_action = Actions.UP if maze.indicator == Indicator_Dir.UP else Actions.DOWN
            obs, r, done, _ = maze.step((correct_action,))
            cur_check = maze.get_cur_check()
            correct_obs_no_check = maze.OBS.END
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=final_intermediate_indicator, at_end=True)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 4.0)
            self.assertTrue(done)

    def testIntermediateIndicatorsNoReset(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["intermediate_checks"] = True
        new_conf["intermediate_indicators"] = True
        new_conf["reset_intermediate_indicators"] = False
        new_conf["allow_left"] = False
        maze = TMaze(new_conf)
        prev_checks = maze.checks
        for _ in range(10):
            maze = TMaze(new_conf)
            self.assertTrue(all(maze.checks == prev_checks))
            prev_checks = maze.checks
        for _ in range(10):
            maze.reset()
            self.assertTrue(all(maze.checks == prev_checks))
            prev_checks = maze.checks

    def testResetPerStep(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["intermediate_checks"] = True
        new_conf["intermediate_indicators"] = True
        new_conf["reset_intermediate_indicators"] = True
        new_conf["per_step_reset"] = True
        new_conf["num_indicators_components"] = n = 10
        maze = TMaze(new_conf)
        for per_step_reset in [True, False]:
            obs = maze.reset()
            maze.per_step_reset = per_step_reset
            prev_obs = None
            obs_changed = False
            for _ in range(100):
                for action in [Actions.LEFT, Actions.RIGHT]:
                    obs, r, done, _ = maze.step(a_2_tuple_a(maze, action, n, obs, wrong=True))
                    if prev_obs is not None and prev_obs != obs:
                        obs_changed = True
                    prev_obs = obs
            if per_step_reset:
                self.assertTrue(obs_changed)
            else:
                self.assertFalse(obs_changed)
        maze.per_step_reset = True
        first_state_check_obs_changed = False
        for _ in range(100):
            first_state_obs = maze.reset()
            self.assertEqual(maze.cur_pos, 0)
            obs, _, _, _ = maze.step(a_2_tuple_a(maze, Actions.UP, n, first_state_obs))
            self.assertEqual(maze.cur_pos, 1)
            new_first_state_obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.LEFT, n, obs))
            self.assertEqual(maze.cur_pos, 0)
            if first_state_obs[3:] != new_first_state_obs[3:]:
                first_state_check_obs_changed = True
        self.assertTrue(first_state_check_obs_changed)

    def testTaskSwitch(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["task_switch_after"] = 500
        new_conf["num_indicators_components"] = n = 10
        maze = TMaze(new_conf)
        for final_action in [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN]:
            # Go to end by answering checks correctly
            # Walk to second-to-last step
            obs = maze.reset()
            for _ in range(98):
                obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.LEFT, n, obs)) # first action is dummy action, but good to ensure cant go left
                correct_obs_no_check = maze.OBS.MIDDLE
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=False, at_end=False)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0.1)
                self.assertFalse(done)
            # walk to last step
            correct_action = a_2_tuple_a(maze, Actions.UP, n, obs)
            obs, r, done, _ = maze.step(correct_action)
            correct_obs_no_check = maze.OBS.END
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=False, at_end=True)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0.1)
            self.assertFalse(done)
            # Take correct action (w.r.t. long term and intermediate indicator), and make sure it times out instead of giving any reward
            maze.Indicator_Dir = Indicator_Dir.UP
            correct_long_term_a = Actions.UP if maze.indicator == Indicator_Dir.UP else Actions.DOWN
            obs, r, done, _ = maze.step(a_2_tuple_a(maze, correct_long_term_a, n, obs))
            correct_obs_no_check = maze.OBS.END
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertTrue(done)
            maze.reset()
        maze.global_vars["timestep"] = 600
        maze.reset()
        for correct_at_end in [True, False]:
            # Make sure there is noise, and forced to the right through it, regardless of action
            for i in range(98):
                a = i%4
                obs, r, done, _ = maze.step(a_2_tuple_a(maze, a, n))
                correct_obs_no_check = maze.OBS.MIDDLE
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=False, at_end=False)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            # walk to last step and stay there for a bit to make sure Timeout higher
            for _ in range(20):
                obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.LEFT, n, obs))
                correct_obs_no_check = maze.OBS.END
                correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=False, at_end=True)
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            # Take correct or incorect action at end based on loop
            if correct_at_end:
                action = Actions.UP if maze.indicator == Indicator_Dir.UP else Actions.DOWN
            else:
                action = Actions.UP if maze.indicator == Indicator_Dir.DOWN else Actions.DOWN
            obs, r, done, _ = maze.step(a_2_tuple_a(maze, action, n, obs))
            cur_check = maze.get_cur_check()
            correct_obs_no_check = maze.OBS.END
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=False, at_end=True)
            self.assertEqual(obs, correct_obs)
            correct_r = 4.0 if correct_at_end else -0.1
            self.assertEqual(r, correct_r)
            self.assertTrue(done)
            maze.reset()

    def testRandomProgression(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["intermediate_checks"] = True
        new_conf["intermediate_indicators"] = True
        new_conf["allow_left"] = False
        new_conf["num_indicators_components"] = n = 1
        maze = TMaze(new_conf)
        maze.per_step_reset = True
        # Make sure you eventually get to end spamming up, and seperatly spamming down
        for check_action in [Actions.CHECK_UP, Actions.CHECK_DOWN]:
            maze.reset()
            for _ in range(4000):
                maze.step((Actions.UP, check_action))
            self.assertEqual(maze.cur_pos, 99)
        maze.per_step_reset = False
        # Make sure you eventually get to end spamming up then down (for check action)
        maze.reset()
        for _ in range(200):
            maze.step((Actions.UP, Actions.CHECK_UP))
            maze.step((Actions.UP, Actions.CHECK_DOWN))
        self.assertEqual(maze.cur_pos, 99)

    def testRewardPerComponent(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["intermediate_checks"] = True
        new_conf["intermediate_indicators"] = True
        new_conf["allow_left"] = False
        new_conf["num_indicators_components"] = n = 10
        new_conf["frac_correct_components_for_check"] = 0.75
        new_conf["reward_per_correct_component"] = True
        maze = TMaze(new_conf)
        obs = maze.reset()
        right_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=False)
        wrong_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=True)
        mixed_action = wrong_action[:-1] + right_action[-1:]
        obs, r, _, _ = maze.step(mixed_action)
        self.assertEqual(r, 0)
        right_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=False)
        wrong_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=True)
        mixed_action = right_action[:-1] + wrong_action[-1:]
        obs, r, _, _ = maze.step(mixed_action)
        self.assertEqual(r, .1*.9)
        self.assertEqual(maze.cur_pos, 1)
        right_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=False)
        wrong_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=True)
        mixed_action = right_action[:-2] + wrong_action[-2:-1] + right_action[-1:]
        obs, r, _, _ = maze.step(mixed_action)
        self.assertEqual(r, .1*.9)
        self.assertEqual(maze.cur_pos, 2)
        for num_incorrect in range(10):
            right_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=False)
            wrong_action = a_2_tuple_a(maze, Actions.UP, n, obs, wrong=True)
            mixed_action = right_action[:n-num_incorrect+1] + wrong_action[n-num_incorrect+1:]
            obs, r, _, _ = maze.step(mixed_action)
            num_correct = n-num_incorrect
            frac_correct = (n-num_incorrect)/n
            self.assertEqual(r, 0 if num_correct < 8 else .1*frac_correct)

    def testIntermediateChecks(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["intermediate_checks"] = True
        new_conf["intermediate_indicators"] = True
        new_conf["allow_left"] = False
        new_conf["num_indicators_components"] = n = 10
        maze = TMaze(new_conf)
        obs = maze.reset()
        # Try left all directions but wrong check actions and make sure you dont move
        for action in [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]:
            obs, r, done, _ = maze.step(a_2_tuple_a(maze, action, n, obs, wrong=True))
            correct_obs_no_check = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                                   maze.OBS.START_DOWN 
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, maze.get_cur_check(), n,
                                                    final_intermediate_indicator=True, at_end=False)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
        cur_check = maze.get_cur_check()
        # Try correct action
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.DOWN, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.MIDDLE
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0.1)
        self.assertFalse(done)
        # Try incorrect action in Middle
        incorrect_action = a_2_tuple_a(maze, Actions.DOWN, n, obs, wrong=True)
        obs, r, done, _ = maze.step(incorrect_action) # This should NOT update cur_check
        correct_obs_no_check = maze.OBS.MIDDLE
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertFalse(done)
        # Try left and right
        for action in [Actions.LEFT, Actions.RIGHT]:
            obs, r, done, _ = maze.step(a_2_tuple_a(maze, action, n, obs, wrong=True))
            correct_obs_no_check = maze.OBS.MIDDLE
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=False, at_end=False)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
        # Allow left and try left back to start
        maze.allow_left = True
        action = Actions.LEFT
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, action, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                               maze.OBS.START_DOWN
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertFalse(done)
        maze.allow_left = False
        # Place indicator at next step and take one step to it correctly
        maze.indicator_pos = 1
        obs = maze.reset()
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.START
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
        self.assertEqual(obs, correct_obs)
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.UP, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.MIDDLE_UP if maze.indicator == Indicator_Dir.UP else \
                               maze.OBS.MIDDLE_DOWN
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0.1)
        self.assertFalse(done)
        # Walk to second-to-last step
        cur_check = maze.get_cur_check()
        for _ in range(97):
            obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.RIGHT, n, obs))
            cur_check = maze.get_cur_check()
            correct_obs_no_check = maze.OBS.MIDDLE
            correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=False)
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0.1)
            self.assertFalse(done)
        # walk to last step
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, Actions.DOWN, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.END
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=True)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0.1)
        self.assertFalse(done)
        # Try incorrect action (w.r.t. the indicator from beginning) and correct wrt intermediate indicator
        incorrect_action = Actions.UP if maze.indicator == Indicator_Dir.DOWN else Actions.DOWN
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, incorrect_action, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.END
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=True)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertTrue(done)
        old_checks = maze.checks
        # Reset, walk to end, try correct action (w.r.t. the indicator from beginning)
        maze.reset()
        self.assertFalse((maze.checks == old_checks).all()) 
        for _ in range(200):
            obs, _, _, _ = maze.step(a_2_tuple_a(maze, incorrect_action, n, obs))
        correct_action = Actions.UP if maze.indicator == Indicator_Dir.UP else Actions.DOWN
        obs, r, done, _ = maze.step(a_2_tuple_a(maze, correct_action, n, obs))
        cur_check = maze.get_cur_check()
        correct_obs_no_check = maze.OBS.END
        correct_obs = correct_obs_from_obs_without_check(maze, correct_obs_no_check, cur_check, n,
                                                    final_intermediate_indicator=True, at_end=True)
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 4.0)
        self.assertTrue(done)

    def testReset(self):
        maze = TMaze(TMazeTest.DEFAULT_CONFIG)
        maze.step((Actions.RIGHT,))
        maze.reset()
        self.assertEqual(maze.cur_pos, 0)
        self.assertEqual(maze.timestep, 0)
        num_goal_up = 0
        num_goal_down = 0
        for _ in range(1000):
            maze.reset()
            if maze.indicator == Indicator_Dir.UP:
                num_goal_up += 1
            else:
                num_goal_down += 1
        fraction_up = num_goal_up/1000
        fraction_down = num_goal_down/1000
        assert_allclose(fraction_up, fraction_down, atol=0.1)

    def testTimeout(self):
        maze = TMaze(TMazeTest.DEFAULT_CONFIG)
        for _ in range(1000):
            _, r, done, _ = maze.step((Actions.RIGHT,))
            self.assertEqual(r, 0)
            self.assertFalse(done)
        _, r, done, _ = maze.step((Actions.RIGHT,))
        self.assertEqual(r, 0)
        self.assertTrue(done)

    def testForceNonfinalDecision(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["force_right"] = True
        new_conf["allow_left"] = False
        maze = TMaze(new_conf)
        actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
        # Step to right just using one action
        for action in actions:
            obs = maze.reset()
            correct_obs = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.START_DOWN
            self.assertEqual(obs, correct_obs)
            for step in range(98):
                obs, r, done, _ = maze.step((action,))
                self.assertEqual(obs, maze.OBS.MIDDLE)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            obs, r, done, _ = maze.step((action,))
            self.assertEqual(obs, maze.OBS.END)
            self.assertEqual(r, 0)
            self.assertFalse(done)

    def testPositionalEncoding(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["wave_encoding_len"] = None
        new_conf["pos_enc"] = True
        new_conf["timeout"] = 150
        new_conf["intermediate_indicators"] = True
        new_conf["num_indicators_components"] = 1
        maze = TMaze(new_conf)
        for p in range(100):
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            self.assertEqual(len(obs), 1 + len(maze.OBS.START_UP) + 1)
            self.assertEqual(obs[-1], p+1)
        new_conf["wave_encoding_len"] = 3
        maze = TMaze(new_conf)
        for p in range(100):
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            self.assertEqual(len(obs), 1 + len(maze.OBS.START_UP) + 3)
            self.assertEqual(obs[-3:], maze.pos_wave_encoding(p+1, 3, c=150))

    def testCorrelatedIndicator(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["correlated_indicator_pos"] = 98
        new_conf["indicator_pos"] = 1
        maze = TMaze(new_conf)
        for i in range(10):
            obs = maze.reset()
            self.assertEqual(obs, maze.OBS.START)
            # step right (see indicator)
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            correct_obs = maze.OBS.MIDDLE_UP if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.MIDDLE_DOWN
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # walk to just before next indicator
            for _ in range(96):
                obs, r, done, _ = maze.step((Actions.RIGHT,))
                correct_obs = maze.OBS.MIDDLE
                self.assertEqual(obs, correct_obs)
                self.assertEqual(r, 0)
                self.assertFalse(done)
            # step right (should now see secondary indicator)
            action = Actions.RIGHT
            obs, r, done, _ = maze.step((action,))
            correct_obs = maze.OBS.MIDDLE_UP if maze.secondary_indicator == Indicator_Dir.UP else \
                          maze.OBS.MIDDLE_DOWN
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # Step to end
            obs, r, done, _ = maze.step((Actions.RIGHT,))
            correct_obs = maze.OBS.END
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # step up
            obs, r, done, _ = maze.step((Actions.UP,))
            correct_obs = maze.OBS.END
            self.assertEqual(obs, correct_obs)
            correct_r = 4.0 if (maze.indicator, maze.secondary_indicator) == (Indicator_Dir.UP, Indicator_Dir.DOWN) else 0.0
            self.assertEqual(r, correct_r)
            self.assertTrue(done)
        # Make sure distribution is correct
        num_uu, num_ud, num_du, num_dd = 0, 0, 0, 0 # Should be 1/4 the time each
        times_d_corect = 0 # Will go down each time. Should be 3/4 the time
        total_num = 1000
        for i in range(total_num):
            obs = maze.reset()
            pattern = (maze.indicator, maze.secondary_indicator)
            if pattern == (Indicator_Dir.UP, Indicator_Dir.UP):
                num_uu += 1
            elif pattern == (Indicator_Dir.UP, Indicator_Dir.DOWN):
                num_ud += 1
            elif pattern == (Indicator_Dir.DOWN, Indicator_Dir.UP):
                num_du += 1
            elif pattern == (Indicator_Dir.DOWN, Indicator_Dir.DOWN):
                num_dd += 1
            for _ in range(100):
                maze.step((Actions.RIGHT,))
            obs, r, done, _ = maze.step((Actions.DOWN,))
            correct_obs = maze.OBS.END
            self.assertEqual(obs, correct_obs)
            down_patterns = [(Indicator_Dir.UP, Indicator_Dir.UP), (Indicator_Dir.DOWN, Indicator_Dir.UP), (Indicator_Dir.DOWN, Indicator_Dir.DOWN)]
            success = (maze.indicator, maze.secondary_indicator) in down_patterns
            correct_r = 4.0 if success else 0.0
            self.assertEqual(r, correct_r)
            self.assertTrue(done)
            if success:
                times_d_corect += 1
        assert_allclose(num_uu/total_num, 0.25, atol=0.05)
        assert_allclose(num_ud/total_num, 0.25, atol=0.05)
        assert_allclose(num_du/total_num, 0.25, atol=0.05)
        assert_allclose(num_dd/total_num, 0.25, atol=0.05)
        assert_allclose(times_d_corect/total_num, 0.75, atol=0.05)

    def testFlippedIndicator(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["flipped_indicator_pos"] = 3
        new_conf["indicator_pos"] = 1
        maze = TMaze(new_conf)
        for i in range(10):
            obs = maze.reset()
            self.assertEqual(obs, maze.OBS.START)
            # step right (see indicator)
            action = Actions.RIGHT
            obs, r, done, _ = maze.step((action,))
            correct_obs = maze.OBS.MIDDLE_UP if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.MIDDLE_DOWN
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # step right (no indicator)
            action = Actions.RIGHT
            obs, r, done, _ = maze.step((action,))
            correct_obs = maze.OBS.MIDDLE
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # step right (should now see flipped indicator)
            action = Actions.RIGHT
            obs, r, done, _ = maze.step((action,))
            correct_obs = maze.OBS.MIDDLE_DOWN if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.MIDDLE_UP
            self.assertEqual(obs, correct_obs)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            # Walk to end and take correct or incorrect action
            for _ in range(100):
                obs, r, done, _  = maze.step((Actions.RIGHT,))
            self.assertEqual(obs, maze.OBS.END)
            self.assertEqual(r, 0)
            self.assertFalse(done)
            if i%2 == 0: # correct action
                correct_action = Actions.UP if maze.indicator == Indicator_Dir.UP else Actions.DOWN
                obs, r, done, _  = maze.step((correct_action,))
                self.assertEqual(obs, maze.OBS.END)
                self.assertEqual(r, 4)
                self.assertTrue(done)
            else:
                incorrect_action = Actions.UP if maze.indicator == Indicator_Dir.DOWN else Actions.DOWN
                obs, r, done, _  = maze.step((incorrect_action,))
                self.assertEqual(obs, maze.OBS.END)
                self.assertEqual(r, 0)
                self.assertTrue(done)

    def testForceFinalDecision(self):
        new_conf = dict(TMazeTest.DEFAULT_CONFIG)
        new_conf["force_final_decision"] = True
        new_conf["fail_reward"] = -.1
        maze = TMaze(new_conf)
        # Try left
        action = Actions.LEFT
        obs, r, done, _ = maze.step((action,))
        correct_obs = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                      maze.OBS.START_DOWN 
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertFalse(done)
        # Try up
        action = Actions.LEFT
        obs, r, done, _ = maze.step((action,))
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertFalse(done)
        # Try down
        action = Actions.LEFT
        obs, r, done, _ = maze.step((action,))
        self.assertEqual(obs, correct_obs)
        self.assertEqual(r, 0)
        self.assertFalse(done)
        for indicator in [Indicator_Dir.UP, Indicator_Dir.DOWN]:
            for final_action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]:
                # Reset
                obs = maze.reset()
                self.assertTrue(obs in [maze.OBS.START_UP, maze.OBS.START_DOWN])
                # Set indicator dir
                maze.indicator = indicator
                # Walk to right
                for _ in range(98):
                    obs, r, done, _ = maze.step((Actions.RIGHT,))
                    self.assertEqual(obs, maze.OBS.MIDDLE)
                    self.assertEqual(r, 0)
                    self.assertFalse(done)
                obs, r, done, _ = maze.step((Actions.RIGHT,))
                self.assertEqual(maze.cur_pos, 99)
                self.assertEqual(obs, maze.OBS.END)
                self.assertEqual(r, 0)
                self.assertFalse(done)
                # Take final action
                obs, r, done, _ = maze.step((final_action,))
                correct_reward = 4.0 if (indicator == Indicator_Dir.UP and \
                                         (final_action == Actions.LEFT or final_action == Actions.UP)) \
                                     or \
                                        (indicator == Indicator_Dir.DOWN and \
                                         (final_action == Actions.RIGHT or final_action == Actions.DOWN)) \
                                     else -.1
                self.assertEqual(obs, maze.OBS.END)
                self.assertEqual(r, correct_reward)
                self.assertTrue(done)

    def testRewardsAndTermination(self):
        maze = TMaze(TMazeTest.DEFAULT_CONFIG)

        # Move up, down, left, right in start state
        actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
        for action in actions:
            _, r, done, _ = maze.step((action,))
            self.assertEqual(r, 0)
            self.assertFalse(done)
        # Move  up, down, left, right in middle state
        maze.step((Actions.RIGHT,)) # So that "left" will not bring agent back to start
        for action in actions:
            _, r, done, _ = maze.step((action,))
            self.assertEqual(r, 0)
            self.assertFalse(done)

        # Move to the end and evaluate rewards and termination
        # go up, correctly at end:
        maze.reset()
        self.assertEqual(maze.cur_pos, 0)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        maze.indicator = Indicator_Dir.UP
        _, r, done, _ = maze.step((Actions.UP,))
        self.assertEqual(r, 4)
        self.assertTrue(done)
        # go up, incorrectly at end:
        maze.reset()
        self.assertEqual(maze.cur_pos, 0)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        maze.indicator = Indicator_Dir.DOWN
        _, r, done, _ = maze.step((Actions.UP,))
        self.assertEqual(r, 0)
        self.assertTrue(done)
        # go down, correctly at end:
        maze.reset()
        self.assertEqual(maze.cur_pos, 0)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        maze.indicator = Indicator_Dir.DOWN
        _, r, done, _ = maze.step((Actions.DOWN,))
        self.assertEqual(r, 4)
        self.assertTrue(done)
        # go down, incorrectly at end:
        maze.reset()
        self.assertEqual(maze.cur_pos, 0)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        maze.indicator = Indicator_Dir.UP
        _, r, done, _ = maze.step((Actions.DOWN,))
        self.assertEqual(r, 0)
        self.assertTrue(done)

    def testObservations(self):
        # Check observations when I at 0
        maze = TMaze(TMazeTest.DEFAULT_CONFIG)
        correct_start_obs = maze.OBS.START_UP if maze.indicator == Indicator_Dir.UP else \
                            maze.OBS.START_DOWN 
        self.assertEqual(maze.get_obs(), correct_start_obs)
        maze.step((Actions.RIGHT,))
        self.assertEqual(maze.get_obs(), maze.OBS.MIDDLE)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        self.assertEqual(maze.get_obs(), maze.OBS.END)

        # Check observations when I at 49
        I_49_conf = dict(TMazeTest.DEFAULT_CONFIG)
        I_49_conf["indicator_pos"] = 49
        maze = TMaze(I_49_conf)
        self.assertEqual(maze.get_obs(), maze.OBS.START)
        maze.step((Actions.RIGHT,))
        self.assertEqual(maze.get_obs(), maze.OBS.MIDDLE)
        for i in range(48):
            maze.step((Actions.RIGHT,))
        correct_middle_obs = maze.OBS.MIDDLE_UP if maze.indicator == Indicator_Dir.UP else \
                             maze.OBS.MIDDLE_DOWN 
        self.assertEqual(maze.get_obs(), correct_middle_obs)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        self.assertEqual(maze.get_obs(), maze.OBS.END)

        # Check observations when I at end
        I_99_conf = dict(TMazeTest.DEFAULT_CONFIG)
        I_99_conf["indicator_pos"] = 99
        maze = TMaze(I_99_conf)
        self.assertEqual(maze.get_obs(), maze.OBS.START)
        maze.step((Actions.RIGHT,))
        self.assertEqual(maze.get_obs(), maze.OBS.MIDDLE)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        correct_end_obs = maze.OBS.END_UP if maze.indicator == Indicator_Dir.UP else \
                          maze.OBS.END_DOWN 
        self.assertEqual(maze.get_obs(), correct_end_obs)
    
    def testActions(self):
        # Test that actions move you in desired ditections unless wall there
        maze = TMaze(TMazeTest.DEFAULT_CONFIG)
        self.assertEqual(maze.cur_pos, 0)
        maze.step((Actions.LEFT,))
        self.assertEqual(maze.cur_pos, 0)
        maze.step((Actions.RIGHT,))
        self.assertEqual(maze.cur_pos, 1)
        maze.step((Actions.UP,))
        self.assertEqual(maze.cur_pos, 1)
        maze.step((Actions.DOWN,))
        self.assertEqual(maze.cur_pos, 1)
        for _ in range(200):
            maze.step((Actions.RIGHT,))
        self.assertEqual(maze.cur_pos, 99)
        maze.step((Actions.LEFT,))
        self.assertEqual(maze.cur_pos, 98)

        # Test that left does nothing when not allowed
        no_left_conf = dict(TMazeTest.DEFAULT_CONFIG)
        no_left_conf["allow_left"] = False
        maze = TMaze(no_left_conf)
        maze.step((Actions.RIGHT,))
        self.assertEqual(maze.cur_pos, 1)
        maze.step((Actions.LEFT,))
        self.assertEqual(maze.cur_pos, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)

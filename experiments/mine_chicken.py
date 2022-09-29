'''
A visual experiment for long-term memory involving chicken collection. 
This is an environment used in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).

You can walk through this environment youself by running mine_chicken.py.
'''


import argparse
import numpy as np
import time
import random
import subprocess
import socket
import os
import signal
import sys
import atexit
import glob
from contextlib import closing
from PIL import Image
from pathlib import Path
from shutil import copytree, rmtree
from pathlib import Path

# ray imports
from ray.rllib.examples.t_maze import Indicator_Dir
from ray.rllib.examples.mine_maze import file_path_to_numpy_img, write_obs, MALMO_PLATFORM_DIR

# gym imports
import gym
from gym.spaces import Discrete, Box, Tuple
from gym.envs.registration import EnvSpec
from gym.utils import seeding
from gym.spaces import Discrete, Box

MINECHICKEN_ENV_KEY = "minechicken-v0"

# default config for REPL in main
CONFIG = {
    "num_steps_signal": 48,
    "num_steps_no_signal": 96,
    "min_chickens": 16,
    "success_r": 8, 
    "fail_r": -6, 
    "chicken_r": 0.1, 
    "high_res": False,
    "noise": None, # Defualt noise for nousy env is 0.05
}

def mine_maze_mission(agent_start=(0,0), chicken_start=(1,2), resolution=(42, 42), signal="g", lava=False, num_chickens=1):
    ''' 
    Note on correct action at end: 
    Indicator up:   r    -> Go LEFT
    Indicator down: g    -> Go RIGHT
    '''
    assert signal in ["r", "g", None], signal

    extra_draw_commands = []
    draw_indent = "                    \n"

    # erase prior blocks with air
    prior_draw_commands = []
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        for y in [225, 226, 227, 228, 229, 230, 231, 232]:
            for z in range(-3, 6, 1):
                prior_draw_commands.append('''<DrawBlock x="'''+str(x)+'''" y="'''+str(y)+'''" z="'''+str(z)+'''" type="air"/>''')

    # Select block for floor
    if lava:
        block = "lava"
    elif signal is None:
        block = "cobblestone"
    elif signal == "g":
        block = "emerald_block"
    elif signal == "r":
        block = "redstone_block"
    # Draw floor 
    for x in [-1, 0, 1]:
        for z in [0, 1, 2, 3]:
            y = 225
            extra_draw_commands.append('''<DrawBlock x="'''+str(x)+'''" y="'''+str(y)+'''" z="'''+str(z)+'''" type="'''+block+'''"/>''')
    # Draw walls
    block = "cobblestone"
    for x in [-2, -1, 0, 1, 2]:
        for z in [-1, 0, 1, 2, 3, 4]:
            for y in [226]:
                if x > -2 and x < 2 and z > -1 and z < 4:
                    continue
                extra_draw_commands.append('''<DrawBlock x="'''+str(x)+'''" y="'''+str(y)+'''" z="'''+str(z)+'''" type="'''+block+'''"/>''')

    # Draw chicken(s)
    for _ in range(num_chickens):
        extra_draw_commands.append('''<DrawEntity type="Chicken" x="'''+str(.5-chicken_start[0])+'''" y="226" z="'''+str(.5+chicken_start[1])+'''" />''')

    xml = '''<?xml version="1.0" encoding="UTF-8" ?> 
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>I Can't Quite Remember...</Summary>
        </About>
        
        <ServerSection>
            <ServerInitialConditions>
                <AllowSpawning>false</AllowSpawning>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*168:1,41;3;biome_1" />
                <DrawingDecorator>''' + draw_indent + draw_indent.join(prior_draw_commands) \
                    + draw_indent.join(extra_draw_commands) + draw_indent + '''
                </DrawingDecorator>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>Jake</Name>
            <AgentStart>
                <Placement pitch="30" x="'''+str(0.5-agent_start[0])+'''" y="226" z="'''+str(0.5+agent_start[1])+'''"/>
            </AgentStart>
            <AgentHandlers>
                <DiscreteMovementCommands autoJump="true" autoFall="true"/>
                <VideoProducer want_depth="false">
                    <Width>'''+str(resolution[0])+'''</Width>
                    <Height>'''+str(resolution[1])+'''</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

    return xml


def normalize_obs(obs):
    obs = obs.astype(np.float32)
    return (obs - 128)/128


class MineChicken(gym.Env):
    """
    A visual game using images cached from Malmo, in which an agent has to collect chickens and remember the room color.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:
            num_steps_signal (int): The number of steps allowed in the color room. (It makes sense to be 
                                    divisible by 3, since it takes 3 steps to collect a chicken)
            num_steps_no_signal (bool): The number of steps allowed in the no-color room. (It makes sense to be 
                                        divisible by 3, since it takes 3 steps to collect a chicken)
            min_chickens (int): Minimum number of chickens neceesary in order to have a go at guessing the room color
            success_r (float): The reward for going the right way at then end of the maze (left or right) based on the 
                        indicator
            fail_r (float): The reward for going the wrong way at then end of the maze (left or right) based on the 
                     indicator
            check_success_r (float): The reward for going the right way at the intermediate checks (left/right)
            check_fail_r (float): The reward for going the wrong way at the intermediate checks (left/right)
            chicken_r (float): the reward for collecting a chicken
            high_res (bool): Whether or not to render observations in higher-resolution
            noise (float or None): The scale of Gaussian noise to add to the observations (or None)
    """

    def __init__(self, config):
        required_args = set([  
            "num_steps_signal",
            "num_steps_no_signal",
            "min_chickens",
            "success_r", 
            "fail_r", 
            "chicken_r", 
            "high_res",
            "noise",
        ])
        given_args = set(config.keys())
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)

        self.num_steps_signal = config["num_steps_signal"]
        self.num_steps_no_signal = config["num_steps_no_signal"]
        self.min_chickens = config["min_chickens"]
        self.success_r = config["success_r"]
        self.fail_r = config["fail_r"]
        self.chicken_r = config["chicken_r"]
        self.noise = config["noise"] # Magnitude of Gaussian noise to add or None

        self._spec = EnvSpec(MINECHICKEN_ENV_KEY)
        self.indicator_pos = 0 # Need this to be compliant with maze_runner callback

        # Deal with action spaces
        self.action_space = Discrete(5)
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(42,42,3,))

        self.chicken_valid_xy = {(0,1), (1,2), (-1,2)}
        self.agent_valid_xy = {(x, y) for x in [-1, 0, 1] for y in [0, 1, 2, 3]}
        self.agent_valid_xy = self.agent_valid_xy
        self.read_in_imgs(hi_res=config["high_res"])

        self.seed()
        self.reset()

    def reset(self):
        self.step_num = 0
        self.num_chickens = 0
        self.phase = 0 # 0 is chicken grab 1 is decision/terminal
        self.agent_x = 0 # Relative to current room
        self.agent_y = 0
        self.phase = 0 # On upper platform, then in normal maze room, then at end
        self.chicken_x = self.np_random.choice([-1, 1])
        self.chicken_pos = (self.chicken_x, 2)
        self.indicator = self.np_random.choice([Indicator_Dir.UP, Indicator_Dir.DOWN])
        self.original_color = "g" if self.indicator == Indicator_Dir.UP else "r"
        self.cur_color = self.original_color
        self.won = None # None if decision not made yet
        return self.get_obs()

    def read_in_imgs(self, hi_res):
        realtive_base_dir = "mine_chicken_data/1000_resolution" if hi_res else "mine_chicken_data/42_resolution"
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = os.path.join(parent_dir, realtive_base_dir)
        try:
            self.final_img = file_path_to_numpy_img(base_dir + "/final.png")
            self.lava_img = file_path_to_numpy_img(base_dir + "/lava.png")
            self.goal_img = file_path_to_numpy_img(base_dir + "/goal.png")
            self.signal_2_chicPos_2_agentPos_2_img = {} # dict of dict of dict
            for s in ["g", "r", None]: # signal
                self.signal_2_chicPos_2_agentPos_2_img[s] = {}
                for chicken_pos in self.chicken_valid_xy:
                    self.signal_2_chicPos_2_agentPos_2_img[s][chicken_pos] = {}
                    for agent_pos in self.agent_valid_xy:
                        if agent_pos == chicken_pos:
                            continue
                        img_path = os.path.join(base_dir, 
                            "signal={}/chicken_pos={}/{}.png".format(str(s), chicken_pos, agent_pos))
                        img = file_path_to_numpy_img(img_path)
                        self.signal_2_chicPos_2_agentPos_2_img[s][chicken_pos][agent_pos] = img
        except FileNotFoundError as err:
            print(err, "You may need to run script with --pics first")
            exit()

    def get_obs(self):
        if self.phase == 0:
            obs = self.signal_2_chicPos_2_agentPos_2_img[self.cur_color][self.chicken_pos][self.xy()]
        else:
            assert self.phase == 1, self.phase
            if self.won is None:
                obs = self.final_img
            elif self.won:
                obs = self.goal_img
            else:
                obs = self.lava_img
        self.last_unnormed_obs = obs
        obs = normalize_obs(obs)
        if self.noise is not None:
            noise = self.np_random.normal(loc=0.0, scale=self.noise, size=obs.shape)
            obs += noise
            obs = obs.clip(-1, 1)
        self.last_obs = obs
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def xy(self):
        return (self.agent_x, self.agent_y)

    def get_delta_x_delta_y(self, action):
        hit = False
        if action == 0:
            delta_x, delta_y = -1, 0
        elif action == 1:
            delta_x, delta_y =  0, 1
        elif action == 2:
            delta_x, delta_y =  1, 0
        elif action == 4:
            delta_x, delta_y =  0, -1
        else:
            assert action == 3, action 
            delta_x, delta_y =  0, 0
            hit = True

        # Remove dissalowed deltas
        if self.phase == 1:
            delta_y = 0 # Cannot move forard or backward here
        else:
            potential_pos = (self.agent_x+delta_x, self.agent_y+delta_y)
            if potential_pos not in self.agent_valid_xy - {self.chicken_pos}:
                delta_x, delta_y = 0, 0

        return delta_x, delta_y, hit

    def reset_chicken(self):
        if self.chicken_pos[0] == 0:
            self.chicken_pos = (self.np_random.choice([-1, 1]), 2)
        else:
            self.chicken_pos = (0,1)

    def chicken_in_front(self,):
        if self.chicken_pos[0] == self.xy()[0] and self.chicken_pos[1] == (self.xy()[1]+1):
            return True
        return False

    def step(self, action):
        self.step_num += 1
        if self.step_num >= self.num_steps_signal:
            self.cur_color = None

        delta_x, delta_y, hit = self.get_delta_x_delta_y(action)
        if self.phase == 0:
            done = False
            if (self.step_num >= (self.num_steps_signal+self.num_steps_no_signal)):
                if self.num_chickens >= self.min_chickens:
                    self.phase = 1
                else:
                    done = True
            if self.chicken_in_front() and hit:
                reward = self.chicken_r
                self.num_chickens += 1
                self.reset_chicken()
            else:
                reward = 0
                self.agent_x += delta_x
                self.agent_y += delta_y
            return self.get_obs(), reward, done, {}
        
        # Phase 1

        # Note: If you dont want forward or backward to count as a failed attempt in this phase,
        # then add timeout and these two lines:
            # if delta_x == 0: # If you want unlimitted tries... but may want timeout then
            #     return self.get_obs(), 0.0, False, {}

        # Moved left or right
        correct_right = self.original_color == "g" and delta_x > 0
        correct_left = self.original_color == "r" and delta_x < 0
        if correct_right or correct_left:
            # Won
            reward = self.success_r
            self.won = True
            return self.get_obs(), reward, True, {}
        # Lost 
        reward = self.fail_r
        self.won = False
        return self.get_obs(), reward, True, {}


    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', mode
        return self.last_unnormed_obs


def take_pics(args):
    # Take and cache pictures from full env for cached verions of the env.
    # Malmo Import:
    # Put malmo import here since 1) there needs to be a hack 2) we dont want this to be required unless args.full
    # STEP HACK: for some reason import malmoenv must come from download for .step() to work.
    if not os.path.exists(os.path.join(MALMO_PLATFORM_DIR, "Minecraft/build/libs")):
        print("Malmo platform does not exist.")
        print("Please pip install malmoenv==0.0.6 and apt install openjdk-8-jdk, then")
        print("From {}, please run the following commands:".format(REPO_DIR))
        print("rm -rf ./MalmoPlatform")
        print("python -c \"import malmoenv.bootstrap; malmoenv.bootstrap.download()\"")
        print("\n...to test that malmoenv is working, you can see if there are any errors from:")
        print("python -c \"import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)\"")
        exit()
    sys.path.insert(0, os.path.join(MALMO_PLATFORM_DIR, "MalmoEnv")) 
    import malmoenv
    from skimage.transform import resize # may not be installed on cluster Docker image

    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir_1000 = os.path.join(base_dir, "mine_chicken_data/1000_resolution")
    base_dir_42 = os.path.join(base_dir, "mine_chicken_data/42_resolution")

    # Create image for final step from red and green block image already in dir
    def create_final_img(base_dir):
        red_fp = os.path.join(base_dir, "red.png")
        green_fp = os.path.join(base_dir, "green.png")
        assert os.path.exists(red_fp), red_fp
        assert os.path.exists(green_fp), green_fp
        red = file_path_to_numpy_img(red_fp)
        green = file_path_to_numpy_img(green_fp)
        final = red[:] # copy
        replace_start_col = int(final.shape[1]/2)
        final[:, replace_start_col:, :] = green[:, replace_start_col:, :]
        final = normalize_obs(final)
        final_fp = os.path.join(base_dir, "final.png")
        write_obs(final, fp=final_fp)

    create_final_img(base_dir_42)
    create_final_img(base_dir_1000)

    def settup_dirs(filename, base_dir):
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        fp = os.path.join(base_dir, file_name)
        return fp
    def write_obs_from_xml(xml, fp_1000, fp_42, wait=0.5):
        actions = {"movesouth", "moveeast", "movewest", "movenorth", "attack"}
        env = malmoenv.make()
        env.init(xml, args.port, server='127.0.0.1', action_filter=actions)
        obs = env.reset()
        time.sleep(wait)
        obs = env._peek_obs()
        obs = normalize_obs(obs.reshape((1000, 1000, 3)))
        print("fp_1000: ", fp_1000)
        write_obs(obs, fp=fp_1000)
        small_obs = resize(obs, (42,42), anti_aliasing=True)
        write_obs(small_obs, fp=fp_42)
        print("Saved:", fp_1000)
        env.close()
        time.sleep(0.5)
    first_step = True
    for signal in ["g", "r", None]:
        dir1 = "signal="+str(signal)
        for chicken_pos in [(0,1), (1,2), (-1,2)]:
            dir2 = "chicken_pos="+str(chicken_pos)
            for x in [-1,0,1]:
                for y in [0, 1, 2, 3]:
                    pos = (x, y)
                    if pos == chicken_pos:
                        continue
                    
                    file_name = str(pos)+".png"
                    file_dir_1000 = os.path.join(base_dir_1000, dir1, dir2)
                    file_path_1000 = settup_dirs(file_name, file_dir_1000)
                    file_dir_42 = os.path.join(base_dir_42, dir1, dir2)
                    file_path_42 = settup_dirs(file_name, file_dir_42)
                    print("Saving file:", file_path_1000)
                    nc = 0 if y >= 2 else 1
                    xml = mine_maze_mission(agent_start=pos, chicken_start=chicken_pos, 
                                            resolution=(1000, 1000), signal=signal, num_chickens=nc)
                    wait = 2 if first_step else 0.5
                    write_obs_from_xml(xml, file_path_1000, file_path_42, wait=wait)
                    first_step = False
    # For final phase 
    # Note: it may be easier to comment out loops above and do these seperately, given Malmo bugs
    goal_fp_1000 = os.path.join(base_dir_1000, "goal.png")
    goal_fp_42 = os.path.join(base_dir_42, "goal.png")
    xml = mine_maze_mission(agent_start=(0,0), chicken_start=(0,1), 
                            resolution=(1000, 1000), signal=None, num_chickens=200)
    write_obs_from_xml(xml, goal_fp_1000, goal_fp_42, wait=5)
    
    lava_fp_1000 = os.path.join(base_dir_1000, "lava.png")
    lava_fp_42 = os.path.join(base_dir_42, "lava.png")
    xml = mine_maze_mission(agent_start=(0,0), chicken_start=(0,1), 
                            resolution=(1000, 1000), signal=None, lava=True)
    write_obs_from_xml(xml, lava_fp_1000, lava_fp_42)


def env_loop(args):
    env = MineChicken(CONFIG)
    for i in range(args.episodes):
        print("\n\nStarting New Episode " + str(i) + "\n\n")

        obs = env.reset()
        write_obs(obs)
        if not args.no_sleep:
            time.sleep(1)

        done = False
        step = 0
        tot_r = 0
        while not done:
            print("\nOn step:", step)
            print("About to step. What action?")

            if args.random: # rand agent
                a = env.action_space.sample()
                print("randomly selected:", a)
            elif args.solve: # opt agent
                if step < (CONFIG["num_steps_signal"] + CONFIG["num_steps_no_signal"]):
                    #phase 0
                    if step%6 == 0:
                        a = 2 if env.chicken_pos[0] == 1 else 0 #right/left
                    elif step%6 == 1:
                        a = 1 # forward
                    elif step%6 == 2:
                        a = 3 # hit
                    elif step%6 == 3:
                        a = 4 # back
                    elif step%6 == 4:
                        a = 0 if env.xy()[0] == 1 else 2 #left/right
                    elif step%6 == 5:
                        a = 3 # hit
                else:
                    if args.subopt:
                        a = np.random.choice([0,2])
                        print("randomly selected:", a)
                    else:
                        if env.original_color == "g":
                            assert env.indicator == Indicator_Dir.UP
                            # go right
                            a = 2
                        else:
                            a = 0
                            # go left
                        print("optimally selected:", a)
            else: # prompt user for input
                # For convienience, map button left of 1 to 0
                valid_action = False
                while not valid_action:
                    try:
                        print("Forward=1, Right=2, Left=0 or `, Back=4 or q, Attack=3")
                        a_str = input()
                        a_str = "0" if a_str == "`" else a_str # map ` to left
                        a_str = "4" if a_str == "q" else a_str # map q to backwards
                        a = int(a_str)
                        if a not in {0,1,2,3,4}:
                            raise ValueError
                        valid_action = True
                    except ValueError as err:
                        print("\nPlease enter a valid action\n")

            obs, reward, done, info = env.step(a)
            print("stepped")

            if not args.no_sleep:
                write_obs(obs)

            print("reward:", reward)
            print("done:", done)
            print("flat obs:", obs.flatten())

            tot_r += reward
            step += 1
            if not args.no_sleep:
                time.sleep(1)
        tot_r = np.round(tot_r, 5)
        print("\nReward for episode: ", tot_r)
        if args.solve:
            expected_tot_r = int((CONFIG["num_steps_signal"]+CONFIG["num_steps_no_signal"])/3)*CONFIG["chicken_r"]
            if args.subopt: # no memory
                expected_tot_r += 0.5*CONFIG["success_r"] + 0.5*CONFIG["fail_r"]
            else:
                expected_tot_r += CONFIG["success_r"]
            expected_tot_r = np.round(expected_tot_r, 3)
            print("\nExpected Reward for episode: ", expected_tot_r)
            if not args.subopt:
                assert expected_tot_r == tot_r, (expected_tot_r, tot_r)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Interact with MineMaze env')
    parser.add_argument('--pics', 
        action="store_true", 
        default=False, 
        help="Replace cached images for this environment by launching full Malmo env.")
    parser.add_argument('--port', 
        type=int, 
        default=9000, 
        help='Server running for Malmo. (If --pics)')
    parser.add_argument('--episodes', 
        type=int, 
        default=2, 
        help='Number of episodes to run.')
    parser.add_argument('--random', 
        action="store_true", 
        default=False, 
        help='Whether to let a random agent solve the maze.')
    parser.add_argument('--solve', 
        action="store_true", 
        default=False, 
        help='Whether to let an optimal agent solve the maze. (Note: not compatible with args.full currently)')
    parser.add_argument('--subopt', 
        action="store_true", 
        default=False, 
        help='For solve option, whether act at the end as if your memory is order invariant.')
    parser.add_argument('--no_sleep', 
        action="store_true", 
        default=False, 
        help="Don't sleep between steps in loop nor write obs to disk")
    args = parser.parse_args()

    # check args
    if args.solve:
        assert not args.random, "Cannot have --solve and --random"

    # Cache pics for cached env
    if args.pics:
        take_pics(args)
        exit()
    
    # Run environment loop. (Either a REPL or automatically solved.)
    env_loop(args)


if __name__ == '__main__':
    main()
'''
A visual experiment for long-term memory. Defines a Malmo scenario, an env to wrap that scanario with a 
Malmo server, and an faster env that takes saved screenshots from that scenario for running quickly
without a Malmo sever.  

The faster env ("minemaze-v0") is the environment used in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).

You can walk through this environment youself by running mine_maze.py.
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

# gym imports
import gym
from gym.spaces import Discrete, Box, Tuple
from gym.envs.registration import EnvSpec
from gym.utils import seeding
from gym.spaces import Discrete, Box

MINEMAZE_ENV_KEY = "minemaze-v0"
FULL_MINEMAZE_ENV_KEY = "fullminemaze-v0"

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR))))
MALMO_PLATFORM_DIR = os.path.join(REPO_DIR, "MalmoPlatform")

DEVNULL = None

# Default configs for REPL in main()
FULL_CONFIG = {
    "simple_len": 205, 
    "num_rooms": 3, 
    "multi_step_indicator": False, 
    "success_r": 4, 
    "fail_r": -3, 
    "check_success_r": 1, 
    "check_fail_r": -1,
    "reward_per_progress": 0.1,
    "short_recent_mem": False,
    "shape": (1000,1000),
    "allow_back": False,
    "timeout": 200,
}
CACHED_CONFIG = {
    "num_rooms": 16,
    "multi_step_indicator": True,
    "num_single_step_repeats": 1,
    "success_r": 4, 
    "fail_r": -3, 
    "check_success_r": 0.1, 
    "check_fail_r": 0.0,
    "reward_per_progress": 0.1,
    "timeout": 200,
    "high_res": True,
    "noise": 0.05,
}

def find_free_port():
    # Adapted from Stack Overlfow
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _, port = s.getsockname()
        return port

def extra_draws_for_complex(short_recent_mem, num_rooms, multi_step_indicator, indicator_up):
    extra_draw_commands = []
    progress_markers = []
    len_ = num_rooms * 4 + 1

    if multi_step_indicator:
        assert num_rooms >= 2, num_rooms

    # Draw floor
    for x in [-1, 0, 1]:
        for y in [225, 226]:
            extra_draw_commands.append('''<DrawLine x1="'''+str(x)+'''" y1="'''+str(y)+'''" z1="0" x2="'''+str(x)+'''" y2="'''+str(y)+'''" z2="'''+str(len_)+'''" type="bedrock"/>''')

    # Set indicator
    if multi_step_indicator:
        indicator = ("emerald_block", "redstone_block") if indicator_up else \
            random.choice([("redstone_block", "emerald_block"), 
                           ("redstone_block", "redstone_block"), 
                           ("emerald_block", "emerald_block")])
    else:
        indicator = ("emerald_block", None) if indicator_up else ("redstone_block", None)

    # Draw indicator and repeating features
    for room_num in range(num_rooms):
        z = room_num * 4
        # Pick Long term Indicator Block or set to default
        if room_num == 0:
            i_block = indicator[0]
        elif room_num == 1 and multi_step_indicator:
            i_block = indicator[1]
        else:
            i_block = "stone"
        # i_block = "stone" # To override above
        # Pick Short Term Signal and Block
        # sand will be good (as defined in mine_maze_mission())
        # diamond -> sand on right
        # iron -> sand on left
        s = random.choice([True, False])
        s_block = "iron_block" if s else "diamond_block" # defines obs for intermediate signal = true, false
        left_fall_block, right_fall_block = ("sand", "gravel") if s else ("gravel", "sand") # blocks you fall onto for reward
        stair_block = "stone" if i_block == "air" else i_block # make block stairs same color as i
        # Draw blocks
        # draw stairs and platforms
        if room_num != 1 and room_num != 2: # rooms 1 and 2 will have platform drawn as previous room's indicator
            extra_draw_commands.append('''<DrawBlock x="0" y="228" z="'''+str(z)+'''" type="stone"/>''') # decision platform
        extra_draw_commands.append('''<DrawBlock x="1"  y="228" z="'''+str(z+2)+'''" type="stone"/>''')
        extra_draw_commands.append('''<DrawBlock x="-1" y="228" z="'''+str(z+2)+'''" type="stone"/>''')
        extra_draw_commands.append('''<DrawBlock x="-1" y="227" z="'''+str(z+1)+'''" type="stone"/>''') # for step
        extra_draw_commands.append('''<DrawBlock x="1"  y="227" z="'''+str(z+1)+'''" type="stone"/>''') # for step
        # draw cols
        if short_recent_mem:
            left_col_block, right_col_block = "stone", "stone"
        else:
            left_col_block, right_col_block = "iron_ore", "diamond_ore"
        extra_draw_commands.append('''<DrawLine x1="-1" y1="227" z1="'''+str(z+3)+'''" x2="-1" y2="231" z2="'''+str(z+3)+'''" type="'''+left_col_block+'''"/>''')
        extra_draw_commands.append('''<DrawLine x1="1"  y1="227" z1="'''+str(z+3)+'''" x2="1"  y2="231" z2="'''+str(z+3)+'''" type="'''+right_col_block+'''"/>''')
        # draw signal for intermediate task
        extra_draw_commands.append('''<DrawLine x1="0"  y1="227" z1="'''+str(z+1)+'''" x2="0"  y2="231" z2="'''+str(z+1)+'''" type="'''+s_block+'''"/>''')
        # draw blocks you fall onto for reward
        extra_draw_commands.append('''<DrawBlock x="1"  y="226" z="'''+str(z)+'''" type="'''+left_fall_block+'''"/>''') #left
        extra_draw_commands.append('''<DrawBlock x="-1" y="226" z="'''+str(z)+'''" type="'''+right_fall_block+'''"/>''') #right
        # draw indicator for long term task (also as a stair/step)
        extra_draw_commands.append('''<DrawBlock x="0"   y="227" z="'''+str(z+3)+'''" type="'''+i_block+'''"/>''')
        extra_draw_commands.append('''<DrawBlock x="-1"  y="228" z="'''+str(z+3)+'''" type="'''+i_block+'''"/>''')
        extra_draw_commands.append('''<DrawBlock x="1"   y="228" z="'''+str(z+3)+'''" type="'''+i_block+'''"/>''')
        extra_draw_commands.append('''<DrawBlock x="0"   y="228" z="'''+str(z+4)+'''" type="'''+i_block+'''"/>''') # try to overight this block in next room
        # extra_draw_commands.append('''<DrawBlock x="0"   y="229" z="'''+str(z+4)+'''" type="'''+i_block+'''"/>''') # to block next room
        # extra_draw_commands.append('''<DrawBlock x="0"   y="230" z="'''+str(z+4)+'''" type="'''+i_block+'''"/>''') # to block next room
        # extra_draw_commands.append('''<DrawBlock x="0"   y="231" z="'''+str(z+4)+'''" type="'''+i_block+'''"/>''') # to block next room
        # Define progress markers
        progress_markers.extend([(1, 227, z+1), (-1, 227, z+1),
                                 (1, 228, z+2), (-1, 228, z+2),
                                 (0, 226, z+2),
                                 (0, 227, z+3),
                                 (0, 228, z+4),
                                ])

    # Draw blocks on ground at end for termination
    extra_draw_commands.append('''<DrawBlock x="1"   y="226" z="'''+str(len_-1)+'''" type="redstone_ore"/>''')
    extra_draw_commands.append('''<DrawBlock x="-1"  y="226" z="'''+str(len_-1)+'''" type="emerald_ore"/>''')

    return extra_draw_commands, progress_markers, len_


def extra_draws_for_simple(len_, indicator_up):
    extra_draw_commands = []

    # Floor
    extra_draw_commands.append('''<DrawLine x1="0" y1="227" z1="0" x2="0" y2="227" z2="'''+str(len_)+'''" type="stone"/>''')

    # Make room for T at end (delete walls there)
    for x in [1, -1]:
        for y in [228, 229]:
            extra_draw_commands.append('''<DrawBlock x="'''+str(x)+'''"  y="'''+str(y)+'''" z="'''+str(len_-1)+'''" type="air"/>''')

    # Add sides to T
    for x in [2, -2]:
        for y in [228, 229]:
            for z_offset in [0, 1]:
                extra_draw_commands.append('''<DrawBlock x="'''+str(x)+'''"  y="'''+str(y)+'''" z="'''+str(len_-1+z_offset)+'''" type="obsidian"/>''')

    # Add indicator
    i_block = "redstone_block" if indicator_up else "emerald_block"
    for x in [1, -1]:
        for y in [228, 229, 230]:
            extra_draw_commands.append('''<DrawBlock x="'''+str(x)+'''"  y="'''+str(y)+'''" z="1" type="'''+i_block+'''"/>''')

    # Draw blocks on ground at end for termination
    extra_draw_commands.append('''<DrawBlock x="-1" y="227" z="'''+str(len_-1)+'''" type="air"/>''')
    extra_draw_commands.append('''<DrawBlock x="1"  y="227" z="'''+str(len_-1)+'''" type="air"/>''')
    extra_draw_commands.append('''<DrawBlock x="-1" y="226" z="'''+str(len_-1)+'''" type="redstone_ore"/>''')
    extra_draw_commands.append('''<DrawBlock x="1"  y="226" z="'''+str(len_-1)+'''" type="emerald_ore"/>''')

    # Define progress markers
    progress_markers = [(0, 227, z) for z in range(2,len_+1)]

    return extra_draw_commands, progress_markers


def mine_maze_mission(reward_per_progress=0.1, shape=(42, 42), short_recent_mem=False, num_rooms=5, simple_len=20, multi_step_indicator=True, simple=True, success_r=4, fail_r=-3, check_success_r=2, check_fail_r=-2):
    ''' 
    Note on correct action at end: 
    Indicator up:   r [multistep: (r,r), (r,g), (g,g)] -> Go LEFT
    Indicator down: g [multistep: (g,r)]               -> Go RIGHT

    Note on correct action for intermediate checks: 
    iron block:    go LEFT
    diamond block: go RIGHT
    '''

    extra_draw_commands = []
    draw_indent = "                    \n"

    indicator_up = np.random.choice([True, False])

    if simple:
        d, m = extra_draws_for_simple(simple_len, indicator_up)
        len_ = simple_len
        w = 1 # width offset for walls
    else:
        w = 2 # width offset for walls
        d, m, len_ = extra_draws_for_complex(short_recent_mem, num_rooms, multi_step_indicator, indicator_up)
    extra_draw_commands.extend(d)
    progress_markers = ['''<Marker oneshot="true" reward="'''+str(reward_per_progress)+'''" tolerance="1" x="'''+str(x+0.5)+'''" y="'''+str(y+1)+'''" z="'''+str(z+0.5)+'''"/>''' for x,y,z in m]

    # Define rewards for touching stones
    sand_end_r = check_success_r # sand is good
    gravel_end_r = check_fail_r # gravel is bad
    if indicator_up:
        emerald_end_r = success_r
        redstone_end_r = fail_r
    else:
        emerald_end_r = fail_r
        redstone_end_r = success_r

    # erase prior blocks with air
    prior_draw_commands = []
    for x in [-3, -2, -1, 0, 1, 2, 3]:
        for y in [225, 226, 227, 228, 229, 230, 231, 232]:
            for z in range(-3, len_+3, 1):
                prior_draw_commands.append('''<DrawBlock x="'''+str(x)+'''" y="'''+str(y)+'''" z="'''+str(z)+'''" type="air"/>''')

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
                <DrawingDecorator>''' + draw_indent + draw_indent.join(prior_draw_commands) + '''
                    <DrawLine x1="'''+str(w)+'''" y1="227" z1="0" x2="'''+str(w)+'''" y2="227" z2="'''+str(len_)+'''" type="cobblestone"/>   <!-- Draw left wall -->
                    <DrawLine x1="'''+str(w)+'''" y1="228" z1="0" x2="'''+str(w)+'''" y2="228" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(w)+'''" y1="229" z1="0" x2="'''+str(w)+'''" y2="229" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(w)+'''" y1="230" z1="0" x2="'''+str(w)+'''" y2="230" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="227" z1="0" x2="'''+str(-w)+'''" y2="227" z2="'''+str(len_)+'''" type="cobblestone"/> <!-- Draw right wall -->
                    <DrawLine x1="'''+str(-w)+'''" y1="228" z1="0" x2="'''+str(-w)+'''" y2="228" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="229" z1="0" x2="'''+str(-w)+'''" y2="229" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="230" z1="0" x2="'''+str(-w)+'''" y2="230" z2="'''+str(len_)+'''" type="cobblestone"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="227" z1="-1" x2="'''+str(w)+'''" y2="227" z2="-1" type="bedrock"/>                    <!-- Draw rear wall -->
                    <DrawLine x1="'''+str(-w)+'''" y1="228" z1="-1" x2="'''+str(w)+'''" y2="228" z2="-1" type="bedrock"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="229" z1="-1" x2="'''+str(w)+'''" y2="229" z2="-1" type="bedrock"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="230" z1="-1" x2="'''+str(w)+'''" y2="230" z2="-1" type="bedrock"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="227" z1="'''+str(len_)+'''" x2="'''+str(w)+'''" y2="227" z2="'''+str(len_)+'''" type="obsidian"/> <!-- Draw front wall -->
                    <DrawLine x1="'''+str(-w)+'''" y1="228" z1="'''+str(len_)+'''" x2="'''+str(w)+'''" y2="228" z2="'''+str(len_)+'''" type="obsidian"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="229" z1="'''+str(len_)+'''" x2="'''+str(w)+'''" y2="229" z2="'''+str(len_)+'''" type="obsidian"/>
                    <DrawLine x1="'''+str(-w)+'''" y1="230" z1="'''+str(len_)+'''" x2="'''+str(w)+'''" y2="230" z2="'''+str(len_)+'''" type="obsidian"/>'''\
                    + draw_indent.join(extra_draw_commands) + draw_indent + '''
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>Jake</Name>
            <AgentStart>
                <Placement x="0.5" y="229" z="0.5"/>
            </AgentStart>
            <AgentHandlers>
                <DiscreteMovementCommands autoJump="true" autoFall="true"/>
                <RewardForTouchingBlockType>
                    <Block reward="'''+str(sand_end_r)+'''" type="sand" behaviour="oncePerBlock"/>
                    <Block reward="'''+str(gravel_end_r)+'''" type="gravel" behaviour="oncePerBlock"/>
                    <Block reward="'''+str(emerald_end_r)+'''" type="emerald_ore" behaviour="oncePerBlock"/>
                    <Block reward="'''+str(redstone_end_r)+'''" type="redstone_ore" behaviour="oncePerBlock"/>
                </RewardForTouchingBlockType>
                <RewardForReachingPosition>'''\
                    + draw_indent + draw_indent.join(progress_markers) + '''
                </RewardForReachingPosition>
                <AgentQuitFromTouchingBlockType>
                   <Block type="emerald_ore" />
                   <Block type="redstone_ore" />
                </AgentQuitFromTouchingBlockType>
                <VideoProducer want_depth="false">
                    <Width>'''+str(shape[0])+'''</Width>
                    <Height>'''+str(shape[1])+'''</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

    return xml, indicator_up

def file_path_to_numpy_img(fp):
    return np.flip(np.array(Image.open(fp)), axis=0)

def write_obs(obs, fp=None):
    if fp is None:
        fp = 'temp_image.png'
    obs = (obs * 128) + 128 # unnormalize
    obs = np.clip(np.round(obs), 0, 256)
    obs = obs.astype(np.uint8)
    img = Image.fromarray(np.flip(obs, axis=(0)))
    save_path = os.path.join(str(Path.home()), fp)
    img.save(save_path)
    print("Observation Written to {}".format(save_path))

class MineMazeFull(gym.Env):
    """
    Class to wrap a malmoenv created from mine_maze_mission to make compatible with RLlib

    If port is specified, connect to a server on that port already running. Else create a server.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:

            simple (bool): Whether or not to use a simple corridor, or complex maze with rooms and intermediate
                           tasks.
            simple_len (int): The length of the maze in the simple case
            num_rooms (int): The number of rooms in the maze (defines the length in default complex case)
            multi_step_indicator (bool): Whether the indicaor is a single obseravtion or consists of a pattern
                                   of two over two steps (not supported in simple)
            success_r (float): The reward for going the right way at then end of the maze (left or right) based on the 
                        indicator
            fail_r (float): The reward for going the wrong way at then end of the maze (left or right) based on the 
                     indicator
            check_success_r (float): The reward for going the right way at the intermediate checks (left/right)
            check_fail_r (float): The reward for going the wrong way at the intermediate checks (left/right)
            reward_per_progress (float): The reward for going the right way any step pther than the checks
            short_recent_mem (bool): Whether to amke the columns have the same texture so that there is one
                                 state that is not Markov and requires recent memory only
            shape ((int, int)): REsolution for observation
            timeout (int): The maximum number of steps the agent can take before terminating and receiving 0 reward
            high_res (bool): Whether or not to render observations in higher-resolution
    """

    def __init__(self, config, port=None):
        required_args = set([ 
            "simple",
            "simple_len", 
            "num_rooms", 
            "multi_step_indicator", 
            "success_r", 
            "fail_r", 
            "check_success_r", 
            "check_fail_r",
            "reward_per_progress",
            "short_recent_mem",
            "shape",
            "allow_back",
            "timeout",
        ])
        given_args = set(config.keys())
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)
        
        self.config = config
        self.maintain_server = port == None
        self.port = port

        self.reward_per_progress = config["reward_per_progress"]
        self._spec = EnvSpec(FULL_MINEMAZE_ENV_KEY)
        self.env = None

        self.seed()
        if port is None:
            self.create_server()
        self.reset(connect=False) # Wait until actual reset to connect

        # Deal with action spaces
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=config["shape"]+(3,))

    def reset(self, connect=True):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.step_num = 0
        return self.create_env_and_connect(connect=connect, server_relaunchable = self.maintain_server)

    def create_env_and_connect(self, connect=True, server_relaunchable=True, wait_time=300):
        # Define xml
        c = self.config
        xml, indicator_is_up = mine_maze_mission(
            simple=c["simple"],
            simple_len=c["simple_len"], 
            num_rooms=c["num_rooms"], 
            multi_step_indicator=c["multi_step_indicator"], 
            success_r=c["success_r"], 
            fail_r=c["fail_r"], 
            check_success_r=c["check_success_r"], 
            check_fail_r=c["check_fail_r"],
            reward_per_progress=c["reward_per_progress"],
            short_recent_mem=c["short_recent_mem"],
            shape=c["shape"],
        )
        self.indicator = Indicator_Dir.UP if indicator_is_up else Indicator_Dir.DOWN

        # Define allowed actions
        self.actions = {"movesouth", "moveeast", "movewest"}
        allow_back = c["allow_back"]
        if allow_back:
            self.actions.add("movenorth")

        # Try to create and connect
        if connect:
            connected = False
            t = time.time()
            # RECONNECT HACK: Must do this since reconnect built into malmo not actually noticing server coming up
            while not connected:
                if time.time() - t > wait_time:
                    if server_relaunchable:
                        self.launch_new_server()
                        t = time.time()
                    else:
                        raise ConnectionRefusedError("Malmo server is down")
                try:
                    self.env = malmoenv.make()
                    self.env.init(xml, self.port,
                        server='127.0.0.1', action_filter=self.actions)
                    def make_sure_innerEnvCleaned(): # in case self.close doesn't get called
                        print("exiting and closing inner env...")
                        self.env.close()
                    atexit.register(make_sure_innerEnvCleaned)
                    obs = self.env.reset()
                    obs = obs.astype(np.float32)
                    time.sleep(0.1) # part of OBS HACK below: need delay here to waut for update on real obs
                    obs = self.env._peek_obs()
                    obs = obs.astype(np.float32)
                    obs = self.reshape_obs(obs)
                    self.last_unnormed_obs = obs
                    obs = self.normalize_obs(obs)
                    connected = True
                    # Convert inner env action space to simple list; part of OBS HACK below
                    self.env.action_space = self.env.action_space.actions + ["null"]
                except ConnectionRefusedError as e:
                    self.env.close()
                    print(e)
                    time.sleep(5)
                    print("Will try reconnecting...")

        return obs if connect else None

    def launch_new_server(self, launch_file, terminate_old=True, verbose=False):
        if terminate_old:
            self.server_proc.kill()
        self.port = find_free_port()
        stdout = None if verbose else DEVNULL
        self.server_proc = subprocess.Popen([launch_file,"-port", str(self.port), "-env"], 
                                            close_fds=True,
                                            stdout=stdout)
        def make_sure_dead(): # in case self.close doesn't get called
            print("exiting and killing subprocess...")
            self.server_proc.kill()
        atexit.register(make_sure_dead)

    def create_server(self):
        # SERVER HACK: when move to multiple servers, need each one to have its own folder to build from so they don't crash. rm when done.
        # Then must run command "<path>/Minecraft/quickLaunchClient.sh -port 9000 -env" directly, not through MalmoEnv bootstrap

        server_dir = os.path.join(REPO_DIR, "temp_server_files")
        if not os.path.isdir(server_dir):
            os.mkdir(server_dir)
        large_random_str = hex(random.randint(0, 999999999999999))
        self.my_server_dir = os.path.join(server_dir, large_random_str)
        copytree(MALMO_PLATFORM_DIR, self.my_server_dir)
        def make_sure_deleted(): # in case self.close doesn't get called
            print("exiting and removing data...")
            rmtree(self.my_server_dir, ignore_errors=True)
        atexit.register(make_sure_deleted)
        launch_file = os.path.join(self.my_server_dir, "Minecraft/launchClient.sh")
        self.launch_new_server(launch_file, terminate_old=False)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reshape_obs(self, obs):
        return obs.reshape(*self.observation_space.shape)

    def normalize_obs(self, obs):
        # NOTE: Default data type out of malmoenv is uint8. Please only pass in int or float as obs.
        return (obs - 128)/128

    def step(self, action, verbose=False):
        if self.config["timeout"] is not None and self.step_num >= self.config["timeout"]:
            return np.zeros_like(self.last_obs), 0.0, True, {}

        obs, reward, done, info = self.env.step(a)
        obs = obs.astype(np.float32)

        # OBS HACK: sometimes obs and reward returned before before env actually updated... redo with null aciton
        if not done: 
            time.sleep(0.2) # 0.2 seems to work fairly reliably, but is a bit slow. (Need to allow time for fall)
            r1 = reward
            if verbose:
                print("redoing action with null")
                print("r1:", r1)
            obs, r2, done, info = self.env.step(-1) # Try again with "null" action, in case obs staying the same was a mistake
            obs = obs.astype(np.float32)
            if verbose: print("r2:", r2)
            if r1 == r2:
                reward = r1
            else:
                if verbose: print("r1 != r2 assuming non-zero and not reward_per_progress takes prescedece")
                if r1 == 0:
                    reward = r2
                elif r2 == 0:
                    reward = r1
                elif r1 == self.reward_per_progress:
                    reward = r2
                elif r2 == self.reward_per_progress:
                    reward = r1
                else:
                    reward = r2

        # R HACK: malmo is return rewards with small decimal added on
        reward = np.round(reward, 3)

        if done:
            obs = np.zeros_like(self.last_obs)
        else:
            obs = self.reshape_obs(obs)
            self.last_unnormed_obs = obs
            obs = self.normalize_obs(obs)
            self.last_obs = obs
            self.step_num += 1

        return obs, reward, done, info

    def close(self):
        self.env.close()
        if self.maintain_server:
            self.server_proc.terminate()
            rmtree(self.my_server_dir)

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', mode
        return self.last_unnormed_obs



class MineMaze(gym.Env):
    """
    A visual maze env that is a simplified version of MineMazeFull, such that it saves stored images and 
    doesn't actually need to run Minecraft.

    Args:
        config (gym.envs.registration.EnvSpec): A specification for this env, containing the following:
            num_rooms (int): The number of rooms in the maze (defines the length)
            multi_step_indicator (bool): Whether the indicaor is a single obseravtion or consists of a pattern
                                   of two over two steps
            num_single_step_repeats (int): If single step indicator, the number of repeats of the single observation
            success_r (float): The reward for going the right way at then end of the maze (left or right) based on the 
                        indicator
            fail_r (float): The reward for going the wrong way at then end of the maze (left or right) based on the 
                     indicator
            check_success_r (float): The reward for going the right way at the intermediate checks (left/right)
            check_fail_r (float): The reward for going the wrong way at the intermediate checks (left/right)
            reward_per_progress (float): The reward for going the right way any step pther than the checks
            timeout (int): The maximum number of steps the agent can take before terminating and receiving 0 reward
            high_res (bool): Whether or not to render observations in higher-resolution
            noise (float or None): The scale of Gaussian noise to add to the observations (or None)
    """

    def __init__(self, config):
        required_args = set([  
            "num_rooms", 
            "multi_step_indicator", 
            "num_single_step_repeats",
            "success_r",
            "fail_r",
            "check_success_r",
            "check_fail_r",
            "reward_per_progress",
            "timeout",
            "high_res",
            "noise",
        ])
        given_args = set(config.keys())
        assert given_args == required_args, "Errors on: {}".format(given_args ^ required_args)

        self.num_rooms = config["num_rooms"]
        self.multi_step_indicator = config["multi_step_indicator"]
        self.num_single_step_repeats = config["num_single_step_repeats"]
        self.success_r = config["success_r"]
        self.fail_r = config["fail_r"]
        self.check_success_r = config["check_success_r"]
        self.check_fail_r = config["check_fail_r"]
        self.reward_per_progress = config["reward_per_progress"]
        self.timeout = config["timeout"]
        self.noise = config["noise"] # Magnitude of Gaussian noise to add or None

        self._spec = EnvSpec(MINEMAZE_ENV_KEY)
        self.indicator_pos = 0 # Need this to be compliant with maze_runner callback

        # Deal with action spaces
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(42,42,3,))

        len_phase0 = 2 if self.multi_step_indicator else self.num_single_step_repeats
        self.phase_2_valid_xy = [set([(x, 0) for x in range(len_phase0)]), # phase 0
                                 {(-1, 1), (-1,2), (-1,3), (0,0), (0,1), (0,3), (1,1), (1,2), (1,3)}, #phase 2
                                 {(0, 0)}] # phase 1
        self.read_in_imgs(hi_res=config["high_res"])

        self.seed()
        self.reset()

    def reset(self):
        self.step_num = 1 # Start at one since there is one obs given by this reset. (If timeout is 1, you get this plus terminal obs)
        self.room_num = 0
        self.agent_x = 0 # Relative to current room
        self.agent_y = 0
        self.phase = 0 # On upper platform, then in normal maze room, then at end
        self.room_types = self.np_random.choice([0, 1], size=(self.num_rooms,))
        # if multistep, this makes (r,r),(g,g),(g,r)(r,g) equally likely:
        up_down_probs = [0.25, 0.75] if self.multi_step_indicator else [0.5, 0.5]
        self.indicator = self.np_random.choice([Indicator_Dir.UP, Indicator_Dir.DOWN], p=up_down_probs)
        # Set indicator color for observation.
        # self.x_2_icolor will define both the number of steps in phase 0 (indicator phase) and their color
        if self.multi_step_indicator:
            self.x_2_icolor = ("g", "r") if self.indicator_is_up() else \
                              [("r", "g"), ("g", "g"), ("r", "r")][self.np_random.choice([0,1,2])]
        else:
            color = "g" if self.indicator_is_up() else "r"
            self.x_2_icolor = tuple([color for _ in range(self.num_single_step_repeats)])
        return self.get_obs()

    def cur_room_type(self):
        return self.room_types[self.room_num]

    def read_in_imgs(self, hi_res):
        img_dir = "mine_maze_data/1000_resolution" if hi_res else "mine_maze_data/42_resolution"
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        img_dir = os.path.join(parent_dir, img_dir)
        try:
            green_img = file_path_to_numpy_img(img_dir + "/green.png")
            red_img = file_path_to_numpy_img(img_dir + "/red.png")
            self.color_2_img = {"g": green_img, "r": red_img}
            self.end_img = file_path_to_numpy_img(img_dir + "/end.png")
            room_type_2_img_path = [glob.glob(img_dir + "/room0/" + "*.png"), glob.glob(img_dir + "/room1/" + "*.png")]
            # e.g. [{(0,0): img1, (0,1): img2}, for room type 1
            #       {(0,0): img3, (0,1): img4}] for room type 2
            self.room_type_2_xy_2_img = [{}, {}]
            for room_type in [0,1]:
                for fp in room_type_2_img_path[room_type]:
                    name, ext = os.path.basename(fp).split(".")
                    x, y = [int(coord_str) for coord_str in name.split(",")]
                    assert (x, y) in self.phase_2_valid_xy[1], (x, y)
                    img = file_path_to_numpy_img(fp)
                    self.room_type_2_xy_2_img[room_type][(x,y)] = img
        except FileNotFoundError:
            print("img dir",img_dir,"not found. Please run script with --downsample first")
            exit()

    def indicator_is_up(self):
        return self.indicator == Indicator_Dir.UP

    def get_obs(self):
        if self.phase == 0:
            color = self.x_2_icolor[self.agent_x]
            obs = self.color_2_img[color]
        elif self.phase == 1:
            obs = self.room_type_2_xy_2_img[self.cur_room_type()][self.xy()]
        else:
            assert self.phase == 2, self.phase
            obs = self.end_img
        self.last_unnormed_obs = obs
        obs = self.normalize_obs(obs)
        if self.noise is not None:
            noise = self.np_random.normal(loc=0.0, scale=self.noise, size=obs.shape)
            obs += noise
            obs = obs.clip(-1, 1)
        self.last_obs = obs
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalize_obs(self, obs):
        obs = obs.astype(np.float32)
        return (obs - 128)/128

    def xy(self):
        return (self.agent_x, self.agent_y)

    def get_delta_x_delta_y(self, action):
        if action == 0:
            delta_x, delta_y = -1, 0
        elif action == 1:
            delta_x, delta_y =  0, 1
        else:
            assert action == 2, action
            delta_x, delta_y =  1, 0

        # Take care of movement within a phase
        if self.phase == 0:
            delta_y = 0 # Cannot move forard here
            delta_x = 0 if delta_x == -1 else delta_x # Cannot move left here
        elif self.phase == 1:
            if self.xy() == (0, 0):
                delta_x = 0 # Cannot move l/r
            elif self.xy() == (0, 1):
                delta_y = 0 # Cannot move f
            elif self.xy() == (1, 1) or self.xy() == (-1, 1) or \
                 self.xy() == (1, 2) or self.xy() == (-1, 2):
                delta_x = 0 # Cannot move l/r
            elif self.xy() == (-1, 3):
                delta_y = 0 # Cannot move f
                delta_x = 0 if delta_x == -1 else delta_x # Cannot move left here
            elif self.xy() == (1, 3):
                delta_y = 0 # Cannot move f
                delta_x = 0 if delta_x == 1 else delta_x # Cannot move right here
            elif self.xy() == (0, 3):
                delta_x = 0 # Cannot move l/r 
            else:
                assert self.xy() not in self.phase_2_valid_xy[1], self.xy()
                raise ValueError("Agent pos: "+str(self.xy())+" was not dealt with properly")
        else:
            assert self.phase == 2, self.phase
            delta_y = 0 # Cannot move forard here

        return delta_x, delta_y

    def get_reward_after_pos_update(self, moved):
        '''
        After updating postion update, get the rewardsbased on whether you moved to correct state
        based on intermediate check and long term indicator 
        '''
        reward = 0.0
        if not moved:
            return 0.0
        if self.phase == 2: # long term
            if self.indicator_is_up():
                if self.xy() == (1,0): # Went to right correctly
                    reward = self.success_r
                elif self.xy() == (-1,0): # Went to left incorrectly
                    reward = self.fail_r
            else:
                if self.xy() == (1,0): # Went to right incorrectly
                    reward = self.fail_r
                elif self.xy() == (-1,0): # Went to left correctly
                    reward = self.success_r
        elif self.phase == 1: # intermediate check
            if self.cur_room_type() == 1:
                if self.xy() == (1,1): # Went to right correctly
                    reward = self.check_success_r
                elif self.xy() == (-1,1): # Went to left incorrectly
                    reward = self.check_fail_r
            else:
                assert self.cur_room_type() == 0
                if self.xy() == (1,1): # Went to right incorrectly
                    reward = self.check_fail_r
                elif self.xy() == (-1,1): # Went to left correctly
                    reward = self.check_success_r
        return reward

    def step(self, action):
        if self.timeout is not None and self.step_num >= self.timeout:
            return self.get_obs(), 0.0, True, {}

        reward = 0.0

        delta_x, delta_y = self.get_delta_x_delta_y(action)

        # Deal with rewards for progress (before pos update)
        moved = (delta_x != 0 or delta_y != 0)
        if moved and \
           not self.phase == 2 and \
           not (self.phase == 1 and self.xy() == (0,1)):
            reward += self.reward_per_progress

        # Update agent pos
        self.agent_x += delta_x
        self.agent_y += delta_y

        # Take care of room/phase transitions and done (given pos update)
        done = False
        if self.phase == 0:
            if self.agent_x >= len(self.x_2_icolor):
                self.phase = 1
                self.agent_x, self.agent_y = 0, 0
        elif self.phase == 1:
            if self.agent_y >= 4:
                self.agent_x, self.agent_y = 0, 0
                self.room_num += 1
                if self.room_num >= self.num_rooms:
                    self.phase = 2
        else:
            if self.agent_x != 0:
                done = True # DONE
            assert self.phase == 2, self.phase

        # Deal with rewards for intermediate checks and long term indicator
        reward += self.get_reward_after_pos_update(moved)

        if not done:
            assert self.xy() in self.phase_2_valid_xy[self.phase], self.xy()

        self.step_num += 1

        return self.get_obs(), reward, done, {}

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', mode
        return self.last_unnormed_obs


def env_loop(env, config, args):
    # Run environment
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
                len_phase0 = 2 if config["multi_step_indicator"] else config["num_single_step_repeats"]
                if step < len_phase0: # phase 0
                    a = 2 # Move right at beginning
                elif step >= (len_phase0 + config["num_rooms"]*6): # phase 2 (at the end)
                    correct_a = 2 if env.indicator_is_up() else 0
                    inncorrect_a = 0 if env.indicator_is_up() else 2
                    if args.subopt:
                        correct_incorrect_probs = [0.75, 0.25] if config["multi_step_indicator"] else [0.5, 0.5]
                        a = np.random.choice([correct_a, inncorrect_a], p=correct_incorrect_probs)
                    else: # args.end_strat == "opt"
                        a = correct_a
                else: # phase 1
                    step_in_room = (step-len_phase0)%6
                    if step_in_room == 1:
                        a = 0 if env.cur_room_type()==0 else 2
                    elif step_in_room in [0, 2, 3, 5]:
                        a = 1
                    else:
                        assert step_in_room == 4, step_in_room
                        a = 2 if env.cur_room_type()==0 else 0
                print("optimally selected:", a)
            else: # prompt user for input
                # For convienience, map ` to 0
                print("Forward=1, Right=2, Left=0 or `")
                a_str = input()
                a_str = "0" if a_str == "`" else a_str
                a = int(a_str)

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
            # Print expected total r following optimal policy (Q* with gamma=1)
            num_steps = len_phase0 + config["num_rooms"]*6 + 1
            expected_tot_r = 0
            expected_tot_r += (num_steps-config["num_rooms"]-1)*config["reward_per_progress"]
            expected_tot_r += config["num_rooms"]*config["check_success_r"]
            if args.subopt:
                if config["multi_step_indicator"]:
                    expected_tot_r += (3/4)*config["success_r"] + (1/4)*config["fail_r"]
                else:
                    expected_tot_r += 0.5*config["success_r"] + 0.5*config["fail_r"]
            else:
                expected_tot_r += config["success_r"]
            expected_tot_r = np.round(expected_tot_r, 3)
            print("\nExpected Reward for episode: ", expected_tot_r)
            if not args.subopt:
                assert expected_tot_r == tot_r, (expected_tot_r, tot_r)

    env.close()


def main():
    # NOTE: "full" version where minecraft actually launches is slow and having issues skipping frames/rewards
    # additionally, at least with the random agemt, it is having trouble resetting.
    # The other version uses cached images from Minecraft.
    parser = argparse.ArgumentParser(description='Interact with MineMaze env')
    parser.add_argument('--full', 
        action="store_true", 
        default=False, 
        help='Lauch minecraft for full version, instead of just images.')
    parser.add_argument('--port', 
        type=int, 
        default=None, 
        help='If you have a server running for --full option, specify the server port. Otherwise, a server will be created.')
    parser.add_argument('--episodes', 
        type=int, 
        default=1, 
        help='Number of episodes to run.')
    parser.add_argument('--simple', 
        action="store_true", 
        default=False, 
        help='Whether to run on simple maze. (Currently only for full version.)')
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
    parser.add_argument('--downsample', 
        action="store_true", 
        default=False, 
        help='Instead of interacting, use this script to downsample cached 1000x1000 resolution to 42x42.')
    parser.add_argument('--no_sleep', 
        action="store_true", 
        default=False, 
        help="Don't sleep between steps in loop nor write obs to disk")
    args = parser.parse_args()

    # Checks and cleanup based on args
    if args.simple:
        assert args.full, "Currently simple maze only in full version"
    if args.full:
        assert not args.solve, "Solve currently not supported for full version"
        # Malmo Import:
        # Put malmp inport here since 1) there needs to be a hack 2) we dont want this to be required unless args.full
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
        malmoenv.comms.retry_count = 1 # (part of RECONNECT hack below)
        global DEVNULL
        DEVNULL = open(os.devnull, 'w')
    if args.solve:
        assert not args.random, "Cannot have --solve and --random"

    # Downsample images if args.downsample, then exit
    if args.downsample:
        full_res_dir = "./mine_maze_data/1000_resolution"
        low_res_dir = "./mine_maze_data/42_resolution"
        copytree(full_res_dir, low_res_dir)
        files_to_downsample = glob.glob(os.path.join(low_res_dir, "**/*.png"), recursive=True)
        for fp in files_to_downsample:
            img = Image.open(fp)
            img = img.resize((42,42),Image.ANTIALIAS)
            img.save(fp)
        exit()

    # Create correct env
    if args.full:
        config = FULL_CONFIG
        config["simple"] = args.simple
        env = MineMazeFull(config, port=args.port)
    else:
        config = CACHED_CONFIG
        env = MineMaze(config)
    
    # Register env for interrput signal
    def interruptHandler(sig, frame):
        print("Caught interrupt... Closing env first please...")
        # Make sure to cleanup server when keyboard interrupt happens. 
        # (Although, other callbacks exist for last resort)
        env.close() 
        exit()
    signal.signal(signal.SIGINT, interruptHandler)

    # Run environment loop. (Either a REPL or automatically solved.)
    env_loop(env, config, args)


if __name__ == '__main__':
    main()
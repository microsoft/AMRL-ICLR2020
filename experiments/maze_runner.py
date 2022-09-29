''' 
File for running experiments on TMaze, MineMaze, MineChicken in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
Will write out names of runs (and LSTM states if applicable) in a way consistent with the parsing for
ray.rllib.visualize_runs, ray.rllib.visualize_SNR, ray.rllib.visualize_state 
Note, please use yaml_configs/tmaze/tmaze.yaml or yaml_configs/EXAMPLE__contains_ALL_MODELS dir
for multiple runs (with --folder) as a template. NB the the first two lines of the yaml.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# gym imports
import gym
from gym.spaces import Discrete, Box, Tuple
from gym.envs.registration import EnvSpec
from gym.utils import seeding

# ray imports
import ray
from ray import tune
from ray.rllib.env.async_vector_env import _DUMMY_AGENT_ID
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
# TMaze imports
from ray.rllib.examples.t_maze import TMaze, MultiEnvTMaze, grouped_env_creator, Indicator_Dir
from ray.rllib.examples.t_maze import TMAZE_ENV_KEY
# MineMaze imports
from ray.rllib.examples.mine_maze import MineMaze
from ray.rllib.examples.mine_maze import MINEMAZE_ENV_KEY
# MineMaze imports
from ray.rllib.examples.mine_chicken import MineChicken
from ray.rllib.examples.mine_chicken import MINECHICKEN_ENV_KEY


import os
import csv
import yaml
import json
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from argparse import ArgumentParser
from time import time
from shutil import copy2, rmtree
from pathlib import Path
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
EXTRA_SAVE_DIR_SUFFIX = "_state_and_conf" # Suffix for directory made for extra manual saves, e.g. rnn state and config
RNN_STATE_USER_DATA_KEY = "rnn_states" # key for state storage in episode user data
TIME_USER_DATA_KEY = "end_time" # key for episode completion time storage in episode user data
I_POS_USER_DATA_KEY = "I_pos" # key for indicator position in episode user data
REWARD_HIST_KEY = "agent_rewards_over_ep"
YAML_BASE_DIR = os.path.join(THIS_DIR, "yaml_configs/AMRL/")
ENV_FILE_NAME = "ENV.yaml" # If running with --folder, if this file is present, then all models will share this env config
repo_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR))))
RESULTS_DIR_NAME = "AMRL_results"
DATA_DIR_NAME = "data"
RESULTS_DIR = os.path.join(repo_base_dir, RESULTS_DIR_NAME, DATA_DIR_NAME)
CLUST_RESULTS_DIR = "/data/ray_results/multi-run/clust" # Change based on user needs
GRID_OPT = ["adam"]
OPT_2_GRID_LR = {"adam": [0.005, 0.0005, 0.00005] }
OPT_2_EPS = defaultdict(lambda: None) # Can only be one number or None
TMAZE_STR = "tmaze"
MINEMAZE_STR = "mine_maze"
CHICKEN_STR = "chicken"


def on_episode_start(info, saves):
    """ 
    Init places to save the keys in saves dictionary. Save within episode.user_data.
    """
    episode = info["episode"]
    for ep_key in saves.keys():
        if ep_key == RNN_STATE_USER_DATA_KEY:
            policy = episode._policies[episode.policy_for("single_agent")]
            init_state = policy.get_initial_state()
            episode.user_data[ep_key] = [init_state]
        elif ep_key in [TIME_USER_DATA_KEY, I_POS_USER_DATA_KEY, REWARD_HIST_KEY]:
            episode.user_data[ep_key] = None
        else:
            episode.user_data[ep_key] = []

def on_episode_step(info, saves):
    """ 
    Save the keys in saves dictionary to episode.user_data at every step.
    """
    episode = info["episode"]
    for ep_key in saves.keys():
        if ep_key == RNN_STATE_USER_DATA_KEY:
            # Save rnn state if set
            if _DUMMY_AGENT_ID in episode.agent_has_set_rnn_state:
                episode.user_data[ep_key].append(episode._agent_to_rnn_state[_DUMMY_AGENT_ID])
        elif ep_key in [TIME_USER_DATA_KEY, I_POS_USER_DATA_KEY, REWARD_HIST_KEY]: # Nothing to save per timestep
            continue
        else:
            # Save other saves if pi_info has been saved
            if _DUMMY_AGENT_ID in episode._agent_to_last_pi_info:
                pi_info = episode._agent_to_last_pi_info[_DUMMY_AGENT_ID]
                assert ep_key in pi_info, "Found policy info but not {} inside.".format(ep_key)
                episode.user_data[ep_key].append(pi_info[ep_key])
    
    assert (_DUMMY_AGENT_ID in episode.agent_has_set_rnn_state) == (_DUMMY_AGENT_ID in episode._agent_to_last_pi_info), \
           "Could only retrieve one of state or pi_info."

def on_episode_end(info, saves, gamma):
    """ 
    Save the keys in saves dictionary (now in episode.user_data) to their corresponding directory.
    Filenames will be the episode id with a prefix indicating goal direction.
    Also calculate and save the discounted return as a custom metric.
    """
    episode = info["episode"]
    ep_id = episode.episode_id
    env_vector = info["env"].vector_env
    env = env_vector.envs[info["env_id"]]
    indicator = env.indicator
    if indicator == Indicator_Dir.UP:
        filename_prefix = 'u'
    else:
        assert indicator == Indicator_Dir.DOWN 
        filename_prefix = 'd'
    filename = filename_prefix + str(ep_id)

    episode.user_data[TIME_USER_DATA_KEY] = time() # store time so this can be saved too.
    episode.user_data[I_POS_USER_DATA_KEY] = env.indicator_pos # store indicator position so this can be saved too.
    episode.user_data[REWARD_HIST_KEY] = episode._agent_reward_history[_DUMMY_AGENT_ID]

    for episode_key, directory in saves.items():
        filepath = os.path.join(saves[episode_key], filename)
        array = np.array(episode.user_data[episode_key])
        np.save(filepath, array)

    store_discounted_return_metric(info, gamma)


def store_discounted_return_metric(info, gamma):
    """ Calculate and save discounted return """
    episode = info["episode"]
    rewards = episode._agent_reward_history[_DUMMY_AGENT_ID]
    gamma_powers = gamma**np.array(range(len(rewards)))
    discounted_return = np.sum(rewards*gamma_powers)
    episode.custom_metrics["discounted_return"] = discounted_return


def print_yaml_dict(d, indent_str=""):
    for k, v in d.items():
        if type(v) is dict:
            print(indent_str+k)
            print_yaml_dict(v, indent_str+"    ")
        elif k is "trial_name_creator":
            print(indent_str+"Trial_Name"+":", v(None))
        else:
            print(indent_str+k+":", v)


def prep_for_saving_config_and_callback_data(d, experiment_name, trial_name, save_states, yaml_path, additional_yaml_path=None):
    # save config with results
    experiment_dir = os.path.join(d["local_dir"].replace("~", str(Path.home())), experiment_name)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    state_and_conf_dir = os.path.join(experiment_dir, trial_name+EXTRA_SAVE_DIR_SUFFIX)
    state_and_conf_dir_already_exists = os.path.isdir(state_and_conf_dir)
    if not state_and_conf_dir_already_exists:
        os.mkdir(state_and_conf_dir)
        copy2(yaml_path, state_and_conf_dir)
        if additional_yaml_path is not None:
            copy2(additional_yaml_path, state_and_conf_dir)
    if save_states:
        # Save rnn state
        # Construct a dictionary of (key to save) to (directory name). (Directory name will be changed to full path.)
        # Note that if the values are changed, you will may need to modify: visualize_state.py
        # Note that keys must be RNN_STATE_USER_DATA_KEY, TIME_USER_DATA_KEY, I_POS_USER_DATA_KEY, REWARD_HIST_KEY
        #   or in episode._agent_to_last_pi_info (saved by extra_fetches of policy graph)
        saves = {
            RNN_STATE_USER_DATA_KEY: "state",
            "f_gates": "gate",
            "i_gates": "i_gate",
            "o_gates": "o_gate",
            "j_proposals": "j_proposal",
            TIME_USER_DATA_KEY: "time",
            I_POS_USER_DATA_KEY: "I_pos",
            REWARD_HIST_KEY: "rewards"
        }
        for ep_key, dir_name in saves.items():
            dir_path = os.path.join(state_and_conf_dir, dir_name)
            os.mkdir(dir_path)
            saves[ep_key] = dir_path # Replace name in saves with path
        # Set up callbacks for RNN state
        callbacks = {
            "on_episode_start": tune.function(lambda info: on_episode_start(info, saves)),
            "on_episode_step": tune.function(lambda info: on_episode_step(info, saves)),
            "on_episode_end": tune.function(lambda info: on_episode_end(info, saves, d["config"]["gamma"])),
        }
    else:
        callbacks = {"on_episode_end": tune.function(lambda info: store_discounted_return_metric(info, d["config"]["gamma"]))}
    d["config"]["callbacks"] = callbacks # Setup up callbakcs
    return os.path.normpath(state_and_conf_dir) # Return location for the callback data and yaml file


def get_experiment_dict_from_yaml(
    yaml_path, 
    run_num, 
    env_type="tmaze", 
    clust=False,
    existing_save_dirs=None, 
    just_env_file=None, 
    lr=None, 
    optimizer=None, 
    eps=None, 
    create_dirs=True, 
    return_dict=False,
    require_env=True):
    '''
    Read in this yaml file and make spec for model to run
    Args
        yaml_path:  The path to read in.
        run_num: The run umber to add to the saved experiment name
        env_type: Whether to use TMaze, MineMaze, or MineChicken. One of the strings: tmaze, minemaze, chicken
        clust: Whether running on the cluster 
        existing_save_dirs: A set of paths for save directories of previous (successful) trials, or None
        just_env_file: A special yaml just for the environment config
        lr: A learning rate to override learning rate in yaml
        optimizer: An optimizer to override learning rate in yaml 
        eps: An epsilon for the optimizer to override learning rate in yaml 
        create_dirs: Whether to create a directory and copy config there and set up callbacks. 
                     Set to False if you just want to read the model and will not run experiment.
        return_dict: Whether to return a dict or a spec
        require_env: Whether to require an env config
    Returns
        A Spec or Dict depending on return_dict
    '''
    with open(yaml_path, 'r') as experiment_file:
        try:
            experiment_dict = yaml.load(experiment_file)
        except yaml.YAMLError as err:
            print(err)
    assert len(experiment_dict) == 1, len(experiment_dict) # Only one experiment per yaml
    assert "trial_name" in experiment_dict[list(experiment_dict.keys())[0]]
    experiment_name = list(experiment_dict.keys())[0] # Experiment name is where files are saved. Modified below.
    d = experiment_dict[experiment_name] # d is the spec
    

    # Add env config to this spec if specified seperately in just_env_file
    assert "config" in d
    if just_env_file:
        assert "env_config" not in d["config"], "Cannot have env specified in yaml if specified in just_env_file"
        with open(just_env_file, 'r') as env_file:
            try:
                env_config = yaml.load(env_file)
            except yaml.YAMLError as err:
                print(err)
        assert "env_config" in env_config
        assert len(env_config) == 1, env_config
        env_config = env_config["env_config"] # Only one item in dict. It is called env_config.
        if "dir_name" in env_config:
            experiment_name = env_config["dir_name"] # Overwrite save location if in env_config
            del env_config["dir_name"] # Cannot leave in to be compatible with rllib
        assert not "run_tag" in env_config, "Custom run_tag no longer supported; please save runs in seperate directory."
        if "timesteps_total" in env_config: # replace key for d["stop"]
            d["stop"]["timesteps_total"] = env_config["timesteps_total"]
            del env_config["timesteps_total"]
        for key in ["num_workers", "num_envs_per_worker", "num_gpus"]: # replace keys for d["config"]
            if key in env_config:
                d["config"][key] = env_config[key]
                del env_config[key]
        d["config"]["env_config"] = env_config # Copy over rest of env config
    elif require_env:
        assert "env_config" in d["config"], "Must have just_env_file to specify env or specify in config"

    if create_dirs and (lr is None or optimizer is None): # Will run experiments but not doing grid search; this is probably not what you want for rigorous experimentation
        print("Warning: grid search will not be performed; using lr and optimizer from config...")
        experiment_name = experiment_name + "_EXAMPLE"

    # Parse experiment name and trial name from yaml, and replace trial name string with actual function
    trial_name = d["trial_name"]
    if trial_name == "": # use model name from yaml file as start of trial name
        fn = os.path.basename(yaml_path)
        yaml_ext = ".yaml"
        assert fn[-len(yaml_ext):] == yaml_ext, "Error: yaml {} must end in {}.".format(yaml_path, yaml_ext)
        trial_name = fn[:len(fn)-len(yaml_ext)] # remove extension
    trial_name = trial_name + "_RUN" + str(run_num) # Trial name includes info for run; add the run number if multiple runs
    trial_name = experiment_name + "_model-" + trial_name # Add experiment name to trial name so that it is apparent when running
    if lr is not None or optimizer is not None:
        trial_name = trial_name + ";"
    if optimizer is not None:
        trial_name = trial_name + "opt=" + optimizer
    if lr is not None:
        if trial_name[-1] != ";":
            trial_name = trial_name + ","
        trial_name = trial_name + "lr=" + str(lr)
    d["trial_name_creator"] = tune.function(lambda trial: trial_name)
    del d["trial_name"]

    save = "fetch_lstm_gates" in d["config"] \
            and d["config"]["fetch_lstm_gates"]
    assert "gamma" in d["config"], "Must specify gamma in the config for custom discounted metric"

    # Overwrite lr and optimizer if doing grid search
    if lr is not None:
        d["config"]["lr"] = lr
    if optimizer is not None:
        d["config"]["opt_type"] = optimizer
    if eps is not None:
        d["config"]["epsilon"] = eps

    # Overrite GPU request if no GPU available
    if float(d["config"]["num_gpus"]) > 0:
        if not tf.test.gpu_device_name():
            d["config"]["num_gpus"] = 0
            print("Warning: gpu not available; config has been overwritten to set num_gpus=0.")

    d["local_dir"] = CLUST_RESULTS_DIR if clust else RESULTS_DIR
    if create_dirs:
        # create directories and register callbacks for writing out states and saving config
        state_and_conf_dir = prep_for_saving_config_and_callback_data(d, experiment_name, trial_name, save, yaml_path, additional_yaml_path=just_env_file)
        if (existing_save_dirs is not None) and (state_and_conf_dir in existing_save_dirs):
            print("Warning: state_and_conf_dir ({}) already had a successful trial. Will skip this experiment.".format(state_and_conf_dir))
            return None

    # Add actual TMaze object to d
    run = d["run"]
    if run == "QMIX": # Deal with QMix
        assert env_type == "tmaze", "Only tmaze currently supports multiagent or QMIX"
        # Get num_agents
        assert "num_agents" in d, "num_agents must be specified in yaml"
        num_agents = int(d["num_agents"])
        del d["num_agents"]
        # Multiply stop reward by number of agents
        if "stop" in d and "episode_reward_mean" in d["stop"]:
            d["stop"]["episode_reward_mean"] *= num_agents
        # Construct and register the multi-agent env
        multi_agent_env_creator = grouped_env_creator(num_agents, d["config"]["env_config"])
        TMAZE_MULTI_NAME = "tmaze_multi-v0"
        tune.register_env(TMAZE_MULTI_NAME, multi_agent_env_creator)
        d["env"] = TMAZE_MULTI_NAME
        sample_multi = multi_agent_env_creator(d["config"]["env_config"])
    else:
        assert "num_agents" not in d, "Multi-agent only supported for tmaze and Qmix"
        if env_type == MINEMAZE_STR:
            tune.register_env(MINEMAZE_ENV_KEY, lambda env_config: MineMaze(env_config))
            d["env"] = MINEMAZE_ENV_KEY
        elif env_type == TMAZE_STR:
            tune.register_env(TMAZE_ENV_KEY, lambda env_config: TMaze(env_config))
            d["env"] = TMAZE_ENV_KEY
        else:
            assert env_type == CHICKEN_STR, env_type
            tune.register_env(MINECHICKEN_ENV_KEY, lambda env_config: MineChicken(env_config))
            d["env"] = MINECHICKEN_ENV_KEY

    print("\nUsing config:\n")
    print_yaml_dict(d)
    if return_dict:
        return d
    return tune.Experiment.from_json(experiment_name, d)

def experiment_to_trial_name(experiment):
    return experiment.spec["trial_name_creator"](None)

def get_yaml_paths(env_type, relative_yaml_dir):
    # Get path to single example yaml
    assert env_type in [TMAZE_STR, MINEMAZE_STR, CHICKEN_STR], "argument {} must be in {}".format(env_type, [TMAZE_STR, MINEMAZE_STR, CHICKEN_STR])
    if env_type == TMAZE_STR:
        single_yaml_path = os.path.join(YAML_BASE_DIR, "tmaze_example.yaml")
    elif env_type == MINEMAZE_STR:
        single_yaml_path = os.path.join(YAML_BASE_DIR, "mine_maze_example.yaml")
    else:
        single_yaml_path = os.path.join(YAML_BASE_DIR, "chicken_example.yaml")
    # Get absolute yaml directory path
    if relative_yaml_dir is None:
        yaml_dir = None
    else:
        yaml_dir = os.path.join(YAML_BASE_DIR, relative_yaml_dir)
    return single_yaml_path, yaml_dir 

def change_result_path_to_this_machine(result_path):
    # In case the results were created on another machine, convert to path for this machine. (Must use same filesystem.)
    relative_path = result_path.split(os.path.join(RESULTS_DIR_NAME, DATA_DIR_NAME))[1]
    return os.path.join(RESULTS_DIR, relative_path.strip("/"))

def clean_failed_runs():
    # Find existing failed and successful experiments in RESULTS_DIR so we can resume
    # Find out which trials ray thinks succeeded
    logdirs_successful = set()
    experiment_state_paths = glob(RESULTS_DIR + "/**/experiment_state*.json", recursive=True)
    for state_path in experiment_state_paths:
        with open(state_path) as state_file:
            trials_states = json.load(state_file)['checkpoints']
            for trial_state in trials_states:
                if trial_state["logdir"] is None:
                    continue
                logdir = change_result_path_to_this_machine(trial_state["logdir"])
                if not os.path.isdir(logdir):
                    continue
                state_str = trial_state["status"]
                assert state_str in ["PENDING", "RUNNING", "PAUSED", "ERRORED", "ERROR", "TERMINATED"], state_str
                trial_successful = (state_str == "TERMINATED")
                if trial_successful:
                    logdirs_successful.add(os.path.normpath(logdir))
                # Run an assert to make sure there is a extra save dir (ending in EXTRA_SAVE_DIR_SUFFIX) for this existing logdir and that it was not orphaned.
                # If it was orpahend, then we may accidentally wind up with two logdirs per one save dir, but we need to ensure a bijection.
                assert os.path.isdir(os.path.join(os.path.dirname(logdir), trial_state["trial_name"]+EXTRA_SAVE_DIR_SUFFIX)), \
                    "{} dir does not exist for logdir: {}\nPlease replace it or remove this logdir.".format(EXTRA_SAVE_DIR_SUFFIX, logdir)
    
    # Now see which extra save dirs (created by a run of this script) have a successful associated logdir with trial data from ray. Remove if they do not have one.
    # Use save_dir as key since maze_runner.py creates them before ray runs experiments.
    existing_save_dirs = glob(RESULTS_DIR + "/**/*" + EXTRA_SAVE_DIR_SUFFIX + "*/", recursive=True)
    save_dirs_successful = set()
    for existing_save_dir in existing_save_dirs:
        # Find (hopefully) unique logdir for unique save_dir
        prefix_path = existing_save_dir[ : len(existing_save_dir)-len(EXTRA_SAVE_DIR_SUFFIX)] # Remove EXTRA_SAVE_DIR_SUFFIX
        potential_logdirs = glob(prefix_path + "*/")
        potential_logdirs.remove(existing_save_dir) # Make sure not to include save dir, only other logdir
        # Ensure bijection between logdirs and save dirs
        if len(potential_logdirs) == 0:
            print("Warning: No trial logdir found for {} directory. This can easily happen if the save directory was created by maze_runner.py\
                but the experiment never began or was cancelled. It will be removed.".format(existing_save_dir))
            rmtree(existing_save_dir)
            continue
        assert len(potential_logdirs) <= 1, \
            "For save dir {}, multiple potential logdirs found: {}.\nThere should only be 1. Please remove one and make sure each YAML has a unique directory or trial_name.".format(existing_save_dir, potential_logdirs)
        logdir = os.path.normpath(potential_logdirs[0]) # Found unique logdir  
        logdir_successful = logdir in logdirs_successful # Find out if the trial was successful
        if logdir_successful: # Trial Succeeded
            save_dirs_successful.add(os.path.normpath(existing_save_dir)) # normpath() ensure path string comparison works
        else:                 # Trial Failed
            # remove existing logdir and save_dir and start over
            print("Warning: {} dir had a failed trial. Will remove this experiment.".format(EXTRA_SAVE_DIR_SUFFIX))
            print("Removing directories:\n{}\n{}".format(existing_save_dir, logdir))
            rmtree(existing_save_dir)
            rmtree(logdir)

    return save_dirs_successful


def get_experiments(yaml_path_env_path_tups, save_dirs_successful, args):
    # Convert yaml files to experiments (and prep for saving, callbacks, etc)
    all_experiments = []
    for yaml_path, just_env_file in yaml_path_env_path_tups:
        for run_num in range(args.start_run_num, args.start_run_num+args.num_runs):
            if args.grid:
                for optimizer in GRID_OPT:
                    for lr in OPT_2_GRID_LR[optimizer]:
                        new_experiment = get_experiment_dict_from_yaml(yaml_path, run_num, clust=args.clust, existing_save_dirs=save_dirs_successful,
                            just_env_file=just_env_file, env_type=args.env_type, lr=lr, optimizer=optimizer, eps=OPT_2_EPS[optimizer])
                        if new_experiment is not None:
                            all_experiments.append(new_experiment)
            else:
                new_experiment = get_experiment_dict_from_yaml(yaml_path, run_num, clust=args.clust, existing_save_dirs=save_dirs_successful,
                    env_type=args.env_type, just_env_file=just_env_file)
                if new_experiment is not None:
                    all_experiments.append(new_experiment)

    # Make sure there are experiments
    if not all_experiments:
        print("No new experiments found.")
        exit()

    # Get experiments in shuffled order
    shuffle(all_experiments)
    if args.dnc_first: # Prioritize DNC experiments, sincet they take longer
        dnc_exps = [exp for exp in all_experiments if "dnc" in experiment_to_trial_name(exp).lower()]
        other_exps = [exp for exp in all_experiments if "dnc" not in experiment_to_trial_name(exp).lower()]
        new_order = dnc_exps+other_exps
        assert len(new_order) == len(all_experiments), (new_order, all_experiments)
        assert set([experiment_to_trial_name(exp) for exp in new_order]) == set([experiment_to_trial_name(exp) for exp in all_experiments]), (new_order, all_experiments)
        all_experiments = new_order
    
    # Print experiment info
    exp_names = [experiment_to_trial_name(exp) for exp in all_experiments]
    local_dirs = [exp.name for exp in all_experiments]
    print("\n\nTrial Names: ", exp_names)
    print("\n\nLocal Dirs: ", local_dirs, "\n\n")

    return all_experiments

def get_yaml_env_paths(single_yaml_path, yaml_dir, args):
    # Create list of tuples of (path_to_model_yaml, env_yaml). 
    # The env_yaml can be None, in which case env will be specified in combined model-env yaml.
    yaml_path_env_path_tups = []
    if args.yaml_dir is None:
        yaml_path_env_path_tups = [(single_yaml_path, None)]
    else:
        if not os.path.isdir(yaml_dir):
            print("No experiments dir found. One has been Created at {}.".format(yaml_dir))
            exit()
        _, sub_dirs, files = next(os.walk(yaml_dir))
        assert not files, "Yaml files must all exist in a subdir, each of which can have at most 1 " + ENV_FILE_NAME
        sub_dirs = [os.path.join(yaml_dir, sd) for sd in sub_dirs] # Make absolute
        for sd in sub_dirs:
            just_env_file = None
            yaml_paths = glob(os.path.join(sd, "**/*.yaml"), recursive=True)
            # See if there is a special case yaml for just the env
            for path in yaml_paths:
                file_name = path.split("/")[-1]
                if file_name == ENV_FILE_NAME:
                    assert just_env_file is None, "Cannot have multiple env files per subdir: {}; {}".format(just_env_file, path)
                    just_env_file = path
            if just_env_file is not None:
                yaml_paths.remove(just_env_file)
            just_env_files = [just_env_file for _ in yaml_paths]
            yaml_path_env_path_tups.extend(zip(yaml_paths, just_env_files))
    return yaml_path_env_path_tups


def run_on_cluster(all_experiments, args):
    import time
    # Ray init cluster logic
    if args.no_workers:
        # Just start ray
        ray.init()
    else:
        # wait for workers
        assert args.redis_address is not None, args.redis_address
        ray.init(redis_address=args.redis_address) # Start ray
        args.workers_sync_count is not None
        def _wait_for_workers(worker_count):
            """Wait for all requested number of workers to join the cluster."""
            if worker_count > 0:
                while True:
                    num_nodes = len(ray.global_state.client_table())
                    if num_nodes < worker_count:
                        print("{} workers have joined so far. Waiting for more."
                              .format(num_nodes))
                        time.sleep(10)
                    else:
                        break
        _wait_for_workers(args.workers_sync_count)
        print("\n\nNum_worker_nodes:", args.workers_sync_count, "\n\n")
    # Run
    tune.run_experiments(all_experiments)
    # Ray head wait. (If it finishes, it may be rebooted)
    while True:
        time.sleep(300)
        print("\n\nDONE WITH ALL RUNS\n\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("env_type", type=str, help="{}, {}, or {}".format(TMAZE_STR, MINEMAZE_STR, CHICKEN_STR))
    parser.add_argument("--yaml_dir", type=str, help="yaml directory  (relative to ./yaml_configs/AMRL/) with all models (or None for single example file), e.g. \"DEMO\")", default=None)
    parser.add_argument("--dnc_first", action='store_true', help="Whether to run experients with DNC in the name first, since they take longer", default=False)
    parser.add_argument("--num_runs", type=int, help="Number of runs/seeds per experiment", default=5)
    parser.add_argument("--start_run_num", type=int, help="The number to start counting at for names of runs. (In case you want to add more)", default=1)
    parser.add_argument("--grid", action='store_true', help="Whether to grid search over lr and optimizer defined at top", default=False)
    parser.add_argument("--clust", action='store_true', help="Whether to adapt saving location, etc, for run on cluster", default=False)
    parser.add_argument("--no_workers", action='store_true', help="Whether ignore all workers when running on cluster and just use head node", default=False)
    parser.add_argument("--redis_address", type=str, help="Redis address from start script", default=None)
    parser.add_argument("--workers_sync_count", type=int, help="Wait for given number of workers in ray cluster (includes head)", default=None)
    args = parser.parse_args()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Get absolute path to yaml directoy and example file if no directory is specified.
    # Note: Thw example yamls are not used in AMRL paper; just examples with the model config and env config in same yaml.
    single_yaml_path, yaml_dir = get_yaml_paths(args.env_type, args.yaml_dir)

    # Get yaml files. (Actually, list of tuples of yaml and an overriding env config if specified.)
    yaml_path_env_path_tups = get_yaml_env_paths(single_yaml_path, yaml_dir, args)

    # Clean out failed experiments and get list of successful ones
    save_dirs_successful = clean_failed_runs()

    # Convert yamls to experiments
    all_experiments = get_experiments(yaml_path_env_path_tups, save_dirs_successful, args)

    # Run experiments
    if args.clust:
        run_on_cluster(all_experiments, args)
    else:
        ray.init()
        tune.run_experiments(all_experiments)


if __name__ == "__main__":
    main()
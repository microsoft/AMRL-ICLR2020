'''
A script used to plot the Signal to Noise Ratio (hueristic) over time for untrained models.
Used in AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).

Note: Will also plot gradients over time w.r.t. the first observation, which relies on one function added to lstm model.
File names will represent the names plotted. The yaml files must be compatible with 
ray.rllib.examples.maze_runner.get_experiment_dict_from_yaml()
'''

from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict, OrderedDict
from argparse import ArgumentParser, Namespace
from gym.spaces import Discrete, Box
from pathlib import Path
from glob import glob
import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf

import ray
from ray.rllib.visualize_runs import MODEL_ORDER, COLOR_PALETTE
from ray.rllib.examples.maze_runner import get_experiment_dict_from_yaml
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

import matplotlib as mpl
mpl.use('webagg')
mpl.rcParams['webagg.open_in_browser'] = False
mpl.rcParams['webagg.port'] = '8988'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.weight': 'bold'})
import seaborn as sns
sns.set()

def calculate_SNR(agent_dict, args, agent_num, num_agents):
    SNRs = []
    config = copy.deepcopy(DEFAULT_CONFIG)
    model_defaults = copy.deepcopy(config["model"])
    config.update(agent_dict["config"]["config"]) # This will replace model defaults if not careful
    model_defaults.update(config["model"])
    config["model"] = model_defaults
    obs_space = Box(-1*np.ones(args.input_size), np.ones(args.input_size), dtype=np.float32)
    action_space = Discrete(4) # Any number here is fine, since wont be using actions, but 4 is typical of tmaze
    for i in range(args.num_agents):
        with tf.Session() as sess:
            with tf.variable_scope("agent"+str(i)):
                print("\n\nEvaluating sample of agent {} of {}, named: {} ...\n\n".format(agent_num, num_agents, agent_dict["model_name"]))
                policy = PPOPolicyGraph(obs_space, action_space, config) #PPOAgent(config=agent_dict["config"]["config"])
                SNRs.append(calculate_SNR_for_policy(policy, args))
        tf.reset_default_graph()
    return SNRs

def expected_powers(runs, args):
    runs, steps = zip(*runs)
    for s in steps:
        assert s == steps[0], (s, steps[0]) ## All steps need to be same for calc
    length_of_runs = len(steps[0])
    runs = np.array(runs)
    assert len(runs.shape) == 3, runs.shape
    assert runs.shape[0] == args.num_runs_per_agent, (runs.shape[0], length_of_runs)
    assert runs.shape[1] == length_of_runs, (runs.shape[1], length_of_runs)
    power = np.sum(runs*runs, axis=-1)/(runs.shape[2])
    assert len(power.shape) == 2, power.shape
    assert power.shape[0] == args.num_runs_per_agent, (power.shape[0], args.num_runs_per_agent)
    assert power.shape[1] == length_of_runs, (power.shape[1], length_of_runs)
    expected_power = np.mean(power, axis=0)
    assert expected_power.shape == (length_of_runs,), (expected_power.shape, (length_of_runs,))
    return expected_power, steps[0]

def calculate_SNR_for_policy(policy, args):
    ones_input = np.ones(args.input_size)
    signals, noises = [], []
    for _ in range(args.num_runs_per_agent):
        if not args.noise: # if not just noises, need signals
            signal_observations = np.zeros((args.length_of_decay, args.input_size))
            if args.strong: # signal randomly and repeatedly
                for i in range(len(signal_observations)-int(args.strong_block*len(signal_observations))):
                    if np.random.rand() < args.strong_p:
                        signal_observations[i] = ones_input
            else:
                signal_observations[0] = ones_input # signal at beginning
            signals.append(get_states_from_obs(policy, signal_observations, args.grad))
        if not args.signal: # if not just signals, need noises
            noise_observations = [np.random.choice([1, -1], p=[0.5, 0.5]) * ones_input for _ in range(args.length_of_decay)]
            noises.append(get_states_from_obs(policy, noise_observations, args.grad))
    if args.norm:
        outputs = signals if signals else noises
        runs, steps = zip(*outputs)
        assert len(runs) == 1, len(runs) # Don't want to compute an average over runs
        run, steps = runs[0], steps[0]
        run_norms = np.sqrt(np.sum(run**2, axis=1))
        mean_norm = np.mean(run_norms)
        trajectory = (run_norms - mean_norm)/mean_norm
        to_return = (trajectory, steps)
    elif args.signal:
        to_return = expected_powers(signals, args)
    elif args.noise:
        to_return = expected_powers(noises, args)
    else:
        (s, s_steps), (n, n_steps) = expected_powers(signals, args), expected_powers(noises, args)
        to_return = (s/n, s_steps)
        assert s_steps == n_steps, (s_steps, n_steps)
    return to_return

def get_states_from_obs_slow(policy, observations):
    # Slow and requires change to PPO; also outdate return type
    state = policy.get_initial_state()
    assert len(state) == 2
    outs = []
    for obs in observations:
        action, state, info = policy.compute_single_action(obs, state)
        assert len(state) == 2
        assert "recurrent_out" in info, "Must have recurrent_out as an extra fetch in policy graph"
        assert info["recurrent_out"].shape[0] == 1, info["recurrent_out"].shape[0]
        outs.append(info["recurrent_out"][0])
    return outs 

def get_states_from_obs(policy, observations, grad, grad_step=10): # If grad, return gradients instead
    state = policy.get_initial_state()
    assert len(state) == 2
    fd = {
        policy._is_training: False,
        policy._obs_input: observations,
        policy._seq_lens: np.array([len(observations)])
    }
    fd.update(dict(zip(policy._state_inputs, [[state[0]], [state[1]]])))
    if grad:
        grad_to_run, steps_to_return = policy.model.do1_de0_2_don_de0_tensors(len(observations), step=grad_step)
        do_de0 = policy._sess.run(
            grad_to_run, 
            feed_dict=fd)
        if grad_step == 1:
            assert len(do_de0) == len(observations)
        assert do_de0[0].shape[0] == 1
        vals_to_return = np.array([tensor[0] for tensor in do_de0])
        assert len(vals_to_return.shape) == 2
    else:
        recurrent_out = policy._sess.run(policy.model.recurrent_out, feed_dict=fd)
        assert recurrent_out.shape[0] == 1, recurrent_out.shape
        assert recurrent_out.shape[1] == len(observations), recurrent_out.shape
        vals_to_return = recurrent_out[0]
        steps_to_return = range(len(observations))
    vals_to_return = vals_to_return.astype(np.float64) # Need the precicion soon
    return vals_to_return, tuple(steps_to_return)


def model_name_from_yaml(yaml):
    return (yaml.split("/")[-1]).split(".yaml")[0] # e.g. yaml_dir/AVG.yaml => AVG


def get_all_results(args):
    # Make dict of key=yaml_file, value={"model_name": str, "config": dict, "snr": [snr1, snr2, ...], "steps":[1,2,3...]}, for each yaml
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, "examples/yaml_configs/AMRL")
    yaml_dir = os.path.join(base_dir, args.yaml_dir)
    yaml_to_results_dict = {}
    yaml_files = glob(os.path.join(yaml_dir, "*.yaml"))
    if not yaml_files:
        print("No yaml files found at the top level of", yaml_dir)
        exit()
    exclude_set = set([s for s in args.exclude.split(",") if s != ""])
    yaml_files = [y for y in yaml_files if not model_name_from_yaml(y) in exclude_set]
    for agent_num, yaml in enumerate(yaml_files):
        # Read in from specific yaml at this location 
        config = get_experiment_dict_from_yaml(yaml, 0, create_dirs=False, return_dict=True, require_env=False)
        # Define model name
        model_name = model_name_from_yaml(yaml)
        if args.skip_AMRL and not args.grad and ("AMRL" in model_name and "LSTM" not in model_name):
            continue # ex AMRL-Avg model has same SNR as AVG
        # Make dict for model
        agent_dict = {"model_name": model_name, "config": config, AGENT_KEY: []}
        # Calculate SNR
        if args.dummy:
            steps = [1,2]
            if args.signal:
                agent_dict[AGENT_KEY] = [([1.0,1.1], steps), ([1.1,1.2], steps)]
            elif args.noise:
                agent_dict[AGENT_KEY] = [([0.9,1.0], steps), ([1.1,1.2], steps)]
            else:
                agent_dict[AGENT_KEY] = [([0.8,1.0], steps), ([1.8,1.8], steps)]
        elif args.combo:
            grads = calculate_SNR(agent_dict, args, agent_num=agent_num+1, num_agents=len(yaml_files))
            snr_args = Namespace(**vars(args))
            snr_args.grad = False
            snr_args.signal = False
            snr_args.noise = False
            snrs = calculate_SNR(agent_dict, snr_args, agent_num=agent_num+1, num_agents=len(yaml_files))
            for grad_run, snr_run in zip(grads, snrs):
                grad_vals, grad_steps = grad_run
                snr_vals, snr_steps = snr_run
                assert snr_steps == tuple(range(args.length_of_decay)), (snr_steps, tuple(range(args.length_of_decay)))
                chosen_snr_vals = np.array([snr_vals[i] for i in grad_steps])
                combo_vals = chosen_snr_vals*grad_vals
                agent_dict[AGENT_KEY].append((combo_vals, grad_steps))
        else:
            agent_dict[AGENT_KEY] = calculate_SNR(agent_dict, args, agent_num=agent_num+1, num_agents=len(yaml_files))
        if args.final:
            agent_dict[AGENT_KEY] = [([vals[-1]], [steps[-1]]) for (vals, steps) in agent_dict[AGENT_KEY]]
            if AGENT_KEY == "snr":
                print("final SNRs for agent {}:".format(model_name), agent_dict["snr"], "\n")
        yaml_to_results_dict[yaml] = agent_dict
    return yaml_to_results_dict


def make_plot(yaml_to_results_dict, args):
    fig = plt.figure()
    data = []
    models_to_plot = set()
    # create y_label
    if args.combo:
        y_label = "grad*snr"
    elif args.norm:
        y_label = "(Norm - Mean)/Mean"
    elif args.grad:
        y_label = "grad"
    elif args.signal:
        y_label = "Signal Power"
    elif args.noise:
        y_label = "Noise Power"
    else:
        y_label = "snr"
    # get data in right format
    model_to_final_perfs = defaultdict(list) if args.final else None
    for yaml, agent_dict in yaml_to_results_dict.items():
        assert agent_dict["model_name"] in MODEL_ORDER, (agent_dict["model_name"], MODEL_ORDER)
        models_to_plot.add(agent_dict["model_name"]) 
        for run_num, run in enumerate(agent_dict[AGENT_KEY]):
            v_s_pairs = list(zip(*run))
            if args.final:
                assert len(v_s_pairs) == 1, len(v_s_pairs)
                model_to_final_perfs[agent_dict["model_name"]].append(v_s_pairs[0][0])
            for value, step in v_s_pairs:
                event_dict = {"Model": agent_dict["model_name"], 
                              "Step": step,
                              y_label: value, 
                              "run_num": run_num,
                             }
                data.append(event_dict)
    # create plot
    if args.final:
        plt.xticks(rotation=90)
    # define model order for this plot based on args
    if args.final or args.sort:
        model_order = sorted(model_to_final_perfs.keys(), key=lambda m: np.mean(model_to_final_perfs[m]))
    else:
        model_order = [m for m in MODEL_ORDER if m in models_to_plot]
    model_order = list(OrderedDict.fromkeys(model_order)) # Make sure unique
    # Set and Sum have very similar SNRs, within the margin of error. To make the order standard, you can do:
        # if "SET" in model_order and "SUM" in model_order: # rearrange to put set after sum
        #     new_model_order = []
        #     for m in model_order:
        #         if m == "SET":
        #             continue
        #         new_model_order.append(m)
        #         if m == "SUM":
        #             new_model_order.append("SET")
        #     model_order = new_model_order
    palette = dict(COLOR_PALETTE) # Make a palette with random color for unknown models
    for m in models_to_plot:
        if m not in palette:
            palette[m] = (np.rand(), np.rand(), np.rand())
    if args.final:
        plot = sns.barplot(x="Model", y=y_label, data=pd.DataFrame(data), 
            ci=args.ci, order=model_order, palette=palette)
    else:
        plot = sns.lineplot(x="Step", y=y_label, estimator="mean", hue="Model", 
            hue_order=model_order, data=pd.DataFrame(data), ci=args.ci, palette=palette)
    if not args.no_log:
        plot.set_yscale('log')
    if args.norm:
        title = "Output Noise Norm"
        plot.set_ylim([.25,-.25])
        for text in plot.legend_.texts:
            legend_str = text.get_text()
            legend_str = legend_str.replace("SET", "Average of LSTM Outputs")
            text.set_text(legend_str)
    else:
        if args.signal:
            title = "Signal"
        elif args.noise:
            title = "Noise"
        else:
            title = "SNR"
        if not args.noise: # noise doesnt change in strong vs weak setting
            if args.strong:
                title = title + " Strong"
        if args.combo:
            title = "Gradient and SNR" #"Combo (" + title + ")"
        elif args.grad:
            title = "Gradient (" + title + ")"
        if args.final:
            title = "Final " + title
    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    if args.upper_ylim is not None:
        plot.set_ylim(top=args.upper_ylim)
    if args.title: 
        fig.suptitle(title, fontsize=16)
    fig.set_size_inches(13*0.4, 7*0.4)
    REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    OUT_PATH_PNG = os.path.join(REPO_BASE_DIR, "AMRL_results", args.out_name+".png")
    OUT_PATH_CSV = os.path.join(REPO_BASE_DIR, "AMRL_results", args.out_name+".csv")
    plt.savefig(OUT_PATH_PNG, 
                dpi=700*(1/0.8), bbox_inches="tight")
    if args.final:
        with open(OUT_PATH_CSV, "w+") as file:
            metric_str = title
            file.write("Model;"+metric_str+"\n")
            for k, v in model_to_final_perfs.items():
                performance_str = ",".join([str(metric) for metric in v])
                file.write(str(k)+";"+performance_str+"\n")
    if not args.no_show:
        plt.show()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Should run faster this way, since not batching

    default_exclude = ""
    parser = ArgumentParser()
    parser.add_argument("--yaml_dir", 
        type=str, 
        help="Directory containing the models to visualize, relative to examples/yaml_configs/AMRL/. Only the model definition is used. YAMLs must be at top-level.", 
        default="tmaze/ALL")
    parser.add_argument("--exclude", type=str, help="model names to exclude seperated by commas with no spaces", default=default_exclude)
    parser.add_argument("--extra_exclude", type=str, help="model names to exclude in addition to exclude (which has defaults)", default="")
    parser.add_argument("--ci", type=int, help="confidence interval for plot", default=68)
    parser.add_argument("--num_agents", type=int, help="number of samples (initialized agents) to use when calculating SNR", default=3)
    parser.add_argument("--num_runs_per_agent", type=int, help="number of runs of an agent to estimate a single SNR", default=None)
    parser.add_argument("--input_size", type=int, help="number of dimensions in stimulus", default=4)
    parser.add_argument("--length_of_decay", type=int, help="number of timesteps to feed in no stimulus (or noise)", default=None)
    parser.add_argument("--dummy", action='store_true', help="Whether to return a dummy DNR for testing", default=False)
    parser.add_argument("--signal", action='store_true', help="Whether to display plot of signal over time instead", default=False)
    parser.add_argument("--noise", action='store_true', help="Whether to display plot of noise over time instead", default=False)
    parser.add_argument("--final", action='store_true', help="Whether to plot just final values as a bar plot", default=False)
    parser.add_argument("--strong", action='store_true', help="Whether the signal obs occurs repeatedly and radnomly", default=False)
    parser.add_argument("--strong_p", type=float, help="Probability of a signal per step if strong", default=0.01)
    parser.add_argument("--strong_block", type=float, help="Fraction of steps at end of strnog with no signal", default=.2)
    parser.add_argument("--grad", action='store_true', help="Plot gradients instead of SNR", default=False)
    parser.add_argument("--combo", action='store_true', help="Multiply grad plot by an SNR plot", default=False)
    parser.add_argument("--norm", action='store_true', help="Compute output norm for one run, instead of SNR. Overrides other args.", default=False)
    parser.add_argument("--no_log", action='store_true', help="Do not plot in log scale", default=False)
    parser.add_argument("--no_show", action='store_true', help="Do not show plot on webserver, just save", default=False)
    parser.add_argument("--out_name", type=str, help="filename for output saved in AMRL_results.", default="current_plot")
    parser.add_argument("--lower_ylim", type=float, help="Lower ylim for plot. E.g. 10e-30.", default=None)
    parser.add_argument("--upper_ylim", type=float, help="Upper ylim for plot. E.g. 1.", default=None)
    parser.add_argument("--sort", action='store_true', help="Whether to sort by performance for the legend on lineplots", default=False)
    parser.add_argument("--skip_AMRL", action='store_true', help="Whether to skip models with \"AMRL\" in the name if calculating SNR, since straight through connections have the same SNR as without", default=False)
    parser.add_argument("--title", action='store_true', help="Show title", default=False)

    args = parser.parse_args()

    # Clean up args
    args.exclude = args.exclude + "," + args.extra_exclude
    if args.norm:
        args.noise = True
        args.num_agents = 1
        args.num_runs_per_agent = 1
        args.no_log = True
        args.final = False
    if args.num_runs_per_agent == None: # Defaults
        args.num_runs_per_agent = 1 if args.grad else 5 # Note: 20 in AMRL paper comes from this 5 * 4 (default args.input_size)
    if args.length_of_decay == None: # Defaults
        args.length_of_decay = 2000 if args.strong else 100
    assert not (args.signal and args.noise), "--signal and --noise are mutually exclusive"
    assert not args.grad or (args.noise or args.signal), "--grad must be evaluated in signal or noise env"
    assert not (args.combo and not args.grad), "--combo is a modification of grad plot"

    global AGENT_KEY
    if args.norm:
        AGENT_KEY = "Norm"
    elif args.grad:
        AGENT_KEY = "Gradient"
    elif args.signal:
        AGENT_KEY = "Signal"
    elif args.noise:
        AGENT_KEY = "Noise"
    else:
        AGENT_KEY = "SNR"

    ray.init()

    # Read in models and calculate SNR
    yaml_to_results_dict = get_all_results(args)

    # Plot SNR (or Gradient)
    make_plot(yaml_to_results_dict, args)



if __name__ == "__main__":
    main()
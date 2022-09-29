'''
A script for plotting tensorboard data with multiple runs of a model. Assumes that the dirs
written out for each run were formatted by maze_runner.py. Used for AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
'''

from ray.rllib.examples.maze_runner import EXTRA_SAVE_DIR_SUFFIX
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict, OrderedDict
from argparse import ArgumentParser
from glob import glob
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('webagg')
mpl.rcParams['webagg.open_in_browser'] = False
mpl.rcParams['webagg.port'] = '8988'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.weight': 'bold'})
import seaborn as sns
sns.set()

# Define Model Order of Legend
MODEL_ORDER = [
        "AMRL-Avg",
        "AVG",
        "AMRL-Max", 
        "MAX",
        "SUM",
        "LSTM", 
        "LSTM_STACK",
        "DNC", 
        "SET",
    ]
blue, organge, green, red, purple, brown, pink, grey, yellow = sns.color_palette(n_colors=9)
COLOR_PALETTE = defaultdict(lambda: (0,0,0), {
    "AMRL-Avg": blue,
    "AVG": (blue[0]*1.5, blue[1]*1.5, blue[2]*1.1),
    "AMRL-Max": red,
    "MAX": (red[0]*1.2, red[1]*1.65, red[2]*1.65),
    "SUM": green,
    "LSTM": pink,
    "LSTM_STACK": (purple[0]*1.2, purple[1], purple[2]),
    "DNC": grey,
    "SET": yellow,
})

REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

def run_info_2_dict(run_info):
    # e.g. ["run=1", "lr=0.1", "opt=sgd"] => {"run": "1", "lr": "0.1", "opt": "sgd"}
    to_return = {}
    for s in run_info:
        key, value = s.split("=")
        if key == "opt":
            key = "Optimizer"
        if key == "lr":
            value = float(value)
        to_return[key] = value
    # Currently only supported options:
    assert set(to_return.keys()) == {"run", "lr", "Optimizer"} or set(to_return.keys()) == {"run"}, to_return
    return to_return

def get_model_name_2_runs(args):
    # Make dict of name to [(run_events, run_info), ...] (multiple runs). Note multiple runs share the same name, given multiple seeds.
    RESULTS_DIR = os.path.join(REPO_BASE_DIR, "AMRL_results", "data", args.dir)
    dirs = glob(RESULTS_DIR + "/*/")
    dirs = [d for d in dirs if args.ignore not in d]           # exclude models we want to ignore
    dirs = [d for d in dirs if EXTRA_SAVE_DIR_SUFFIX not in d] # exclude extra directories created by maze_runner.py
    print(dirs)
    print(len(dirs))
    model_name_2_runs = defaultdict(list)
    for d in dirs:
        # e.g. AMRL_results/data/T-L/T-L_model-AVG_RUN1;lr=0.1,opt=sgd_2019-06-12_12-33-476e82ldqy/
        dirname = os.path.basename(os.path.normpath(d)) # => T-L_model-AVG_RUN1;lr=0.1,opt=sgd_2019-06-12_12-33-476e82ldqy
        experiment_and_model_name, run_info = dirname.split("_RUN")    # => T-L_model-AVG     and    1;lr=0.1,opt=sgd_2019-06-12_12-33-476e82ldqy
        model_name = experiment_and_model_name.split("_model-")[1]     # => AVG
        if ";" in run_info: # parse information in run from a grid search (maze_runner.py --grid)
            run_num, run_info = run_info.split(";")
            run_info = run_info.split("_")[0] # e.g. lr=0.1,opt=sgd
            run_info = run_info.split(",") # e.g. ["lr=0.1", "opt=sgd"]
            run_info = ["run="+run_num] + run_info # e.g. ["run=1", "lr=0.1", "opt=sgd"]
        else: # e.g. # 1_2019-06-12_12-33-476e82ldqy
            run_num = run_info.split("_")[0]
            run_info = ["run="+run_num]
        run_info = run_info_2_dict(run_info)
        events = event_accumulator.EventAccumulator(d)
        events.Reload()
        model_name_2_runs[model_name].append((events, run_info))
    return model_name_2_runs

def get_clean_data(model_name_2_runs, args):
    model_to_fullname = {} # fullnames will be made that include some run info
    data = []
    model_2_best_performances = {} # Get metric for best performing learning rate (LR)
    exclude_set = set([s for s in args.exclude.split(",") if s != ""])
    include_set = set([s for s in args.only.split(",") if s != ""])
    for model_name, runs in model_name_2_runs.items():
        legend_name = "Model"
        assert model_name in MODEL_ORDER, "{} not in {}".format(model_name, MODEL_ORDER)
        if model_name in exclude_set:
            continue
        if args.only != "" and model_name not in include_set:
            continue
        num_runs = str(len(runs))
        full_name = (model_name + "_" + str(num_runs) + "runs") if args.nr else model_name # Add number of runs to name in legend
        model_to_fullname[model_name] = full_name
        # Note: if you use a tag that is not the episode length or reward, you should be able to do:
            # y_label = args.tag.split("/")[-1]
        # but it will not be capitalized properly
        y_label = "Episode Length" if "episode_len" in args.tag else "Total Reward"
        x_label = "Step (k)"
        # Find best lr if lr search and calculate run lengths to make sure no events are missing
        max_run_len = 0
        prev_run_info = None
        lr_2_metrics = defaultdict(list) # If multiple lr runs, get final rewards for each run, to find best
        for run, run_info in runs: # runs contains repreats as well as different lr specified in run_info
            if args.tag not in run.Tags()["scalars"]:
                print("Warning: run without given tag")
                continue
            run_len = len(run.Scalars(args.tag))
            max_run_len = max(max_run_len, run_len)
            lr_or_None = run_info["lr"] if "lr" in run_info else None
            if args.final_r:
                metric = run.Scalars(args.tag)[-1].value # Faster than below and works
                # Note, the following is slower but safer:
                    # events = run.Scalars(args.tag)
                    # sorted_events = sorted(events, key=lambda event: event.step)
                    # final_r = sorted_events[-1].value
                    # assert metric == final_r, (metric, final_r)
            elif args.AUC: # Area under the curve
                rs = [event.value for event in run.Scalars(args.tag)]
                steps =[event.step for event in run.Scalars(args.tag)]
                metric = np.trapz(y=rs,x=steps)
            else:
                metric = np.mean([event.value for event in run.Scalars(args.tag)])
            lr_2_metrics[lr_or_None].append(metric)
        best_lr, best_metric = None, None
        for lr, metrics in lr_2_metrics.items():
            avg_metric = np.mean(metrics)
            if best_metric == None or avg_metric > best_metric:
                best_lr = lr
                best_metric = avg_metric
                model_2_best_performances[full_name] = metrics
        print("(Best over lr) Metric for {} is: {}".format(full_name, best_metric))
        # Save data for plotting
        for run, run_info in runs:
            if args.tag not in run.Tags()["scalars"]:
                print("Warning: run without given tag")
                continue
            run_len = len(run.Scalars(args.tag))
            if run_len < max_run_len and (args.step is None or (run.Scalars(args.tag)[-1].step < (args.step*1000))):
                print("Warning, found events missing for a run with {}, run_info {}".format(full_name, run_info))
            if run_len == 0:
                continue
            if args.best and "lr" in run_info and run_info["lr"] != best_lr:
                continue
            for scalar_event in run.Scalars(args.tag):
                if args.step is None or (scalar_event.step <= (args.step*1000)):
                    event_dict = {legend_name: full_name, 
                                  x_label: scalar_event.step/1000, 
                                  y_label: scalar_event.value, 
                                 }
                    event_dict.update(run_info)
                    if "lr" in run_info: # for now, since numberical lr is not displayed correctly
                        event_dict["lr"] = format(run_info["lr"], ".0e")
                    data.append(event_dict)
    if not data:
        print("No data found")
        exit()

    return data, model_2_best_performances, model_to_fullname, x_label, y_label, legend_name

def plot_runs(data, model_2_best_performances, model_to_fullname, x_label, y_label, legend_name, args):
    fig = plt.figure()
    fullname_model_order = [model_to_fullname[m] for m in MODEL_ORDER if m in model_to_fullname]
    estimator = None if args.indiv else "mean"
    ci = None if args.indiv else args.ci
    units = "run" if args.indiv else None
    # Note: To visualize the opimizer used, you may be able to do the following:
        # style = "Optimizer" if "Optimizer" in data[0] else None
        # size = "lr" if "lr" in data[0] else None
    # However, This created issues displaying the lr and AMRL only used Adam optimizer, so we do this instead:
    style = "lr" if ("lr" in data[0]) and (args.lr or not args.best) else None
    size = None
    plot = sns.lineplot(x=x_label, y=y_label, estimator=estimator, units=units, markers=True, 
        style=style, size=size, hue=legend_name, hue_order=fullname_model_order,
        data=pd.DataFrame(data), ci=ci, palette=COLOR_PALETTE)
    
    # clean up text
    for text in plot.legend_.texts:
        legend_str = text.get_text()
        # May have to remove if doing ablation later. 
        # Make this change here since o/w duplicates in model_order cause problems
        truncated = legend_str[:15]
        text.set_text(truncated)

    if args.no_legend:
        plot.get_legend().remove()

    if args.no_ylabel:
        plt.ylabel("")

    if args.lower_ylim is not None:
        plot.set_ylim(bottom=args.lower_ylim)
    
    # save and show plot
    if args.title: 
        fig.suptitle(args.dir, fontsize=16) # Set directory name as title
    fig.set_size_inches(13*0.4, 7*0.4)
    if args.out_name is None:
        OUT_PATH_PNG = '/mnt/jabeckstorage/ray_results/current_plot.png'
        OUT_PATH_CSV = '/mnt/jabeckstorage/ray_results/current_plot.csv'
    else:
        OUT_PATH_PNG = os.path.join(REPO_BASE_DIR, "AMRL_results", args.out_name+".png")
        OUT_PATH_CSV = os.path.join(REPO_BASE_DIR, "AMRL_results", args.out_name+".csv")
    plt.savefig(OUT_PATH_PNG, dpi=700*(1/0.8), bbox_inches="tight")
    with open(OUT_PATH_CSV, "w+") as file:
        if args.final_r:
            metric_str = "Final Return"
        elif args.AUC:
            metric_str = "AUC"
        else:
            metric_str = "AVG_Return"
        file.write("Model;"+metric_str+"\n")
        for k, v in model_2_best_performances.items():
            performance_str = ",".join([str(metric) for metric in v])
            file.write(str(k)+";"+performance_str+"\n")
    if not args.no_show:
        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("dir", type=str, help="the result dir to plot relative to AMRL_results/data/")
    parser.add_argument("--tag", type=str, help="the tensorflow tag to plot", default="ray/tune/smoothed_episode_reward_mean")
    parser.add_argument("--ignore", type=str, help="ignore directories containing this string", default="state_and_conf")
    parser.add_argument("--exclude", type=str, help="model names to exclude seperated by commas with no spaces", default="")
    parser.add_argument("--only", type=str, help="only allow these model names", default="")
    parser.add_argument("--ci", type=int, help="confidence interval for plot", default=68)
    parser.add_argument("--step", type=int, help="max number of steps for plot in thousands (k), or None", default=None)
    parser.add_argument("--indiv", action='store_true', help="Whether to plot individual runs instead of confidence intervals", default=False)
    parser.add_argument("--best", action='store_true', help="Whether to distplay best over lr search (if multiple lr)", default=False)
    parser.add_argument("--nr", action='store_true', help="Include num runs in model legend", default=False)
    parser.add_argument("--lr", action='store_true', help="Show lr in legend if grid search done", default=False)
    parser.add_argument("--final_r", action='store_true', help="Use final reward instead of Average Return as metric for best lr", default=False)
    parser.add_argument("--AUC", action='store_true', help="Use AUC instead of Average Return as metric for best lr", default=False)
    parser.add_argument("--no_show", action='store_true', help="Do not show plot on webserver, just save", default=False)
    parser.add_argument("--no_legend", action='store_true', help="Do not show legend.", default=False)
    parser.add_argument("--no_ylabel", action='store_true', help="Do not show ylabel.", default=False)
    parser.add_argument("--out_name", type=str, help="filename for output. Default only works on my VM.", default=None)
    parser.add_argument("--title", action='store_true', help="Show title", default=False)
    parser.add_argument("--lower_ylim", type=float, help="Lower ylim for plot. E.g. 10e-30.", default=None)
    args = parser.parse_args()

    assert not (args.AUC and args.final_r)

    # Make dict of name to [(run_events, run_info), ...] (multiple runs). Note multiple runs share the same name, given multiple seeds.
    model_name_2_runs = get_model_name_2_runs(args)

    # Clean up data for plotting
    data, model_2_best_performances, model_to_fullname, x_label, y_label, legend_name = get_clean_data(model_name_2_runs, args)

    # Plot
    plot_runs(data, model_2_best_performances, model_to_fullname, x_label, y_label, legend_name, args)



if __name__ == "__main__":
    main()
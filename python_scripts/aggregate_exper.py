'''
Script for summarizing multiple experiments from AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
Combines metrics into overall performance and makes a scatter plot to show
relationship between into Metrics, SNR, and Gradient Decay.
All of these calculations must be done prior to using this script.
See demo_AMRL.sh for an example.
'''

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from subprocess import call
from collections import defaultdict, OrderedDict
import matplotlib as mpl
mpl.use('webagg')
mpl.rcParams['webagg.open_in_browser'] = False
mpl.rcParams['webagg.port'] = '8988'
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from ray.rllib.visualize_runs import MODEL_ORDER, COLOR_PALETTE
sns.set()

REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
RESULTS_DIR = os.path.join(REPO_BASE_DIR, "AMRL_results")
OUTPUT_DIR = RESULTS_DIR
USE_TITLE = False
METRIC_STR = "Mean Return"
ALLOWED_MODELS = {m for m in MODEL_ORDER}


def get_models_2_experiments(experiment_names):    
    # Build up map from models to experiment results, represented as a list of equal-length lists.
    models_2_experiments = defaultdict(list) # each list will contain lists of metrics (one for each run)
    found_models = set()
    num_runs = None
    for name in experiment_names:
        csv_path = os.path.join(RESULTS_DIR, name+".csv")
        print("Incorporating {}...".format(csv_path))
        with open(csv_path, "r") as file:
            lines = file.readlines()
        for line in lines[1:]:
            model, metrics = line.split(";")
            metrics = [float(m) for m in metrics.split(",")]
            assert model in ALLOWED_MODELS, "{} not known model from {}".format(model, csv_path)
            if num_runs is not None:
                assert len(metrics) == num_runs, "Error: all experiments must have same number of runs, but found {} and {}".format(len(metrics), num_runs)
            else:
                num_runs = len(metrics)
            models_2_experiments[model].append(metrics)
            found_models.add(model)

    # Note: To write the mean performance to disk, you should be able to use the following code:
        # out_path = os.path.join(OUTPUT_DIR, "Overall_Performance.csv")
        # with open(out_path, "w+") as out_file:
        #     out_file.write("Model; "+METRIC_STR+"\n")
        #     for model in MODEL_ORDER:
        #         # make sure all have same length
        #         assert len(models_2_experiments[model]) == len(list(models_2_experiments.values())[0]), (len(models_to_metrics[model]), len(list(models_2_experiments.values())[0])
        #         assert len(models_2_experiments[model][0]) == len(list(models_2_experiments.values())[0][0])
        #         # calculate and write mean
        #         mean_metric = np.mean(models_2_experiments[model])
        #         out_file.write(model+"; "+ format(mean_metric, ".3f") + "\n")
        # print("\n\nCSV written to {}.".format(out_path))

    return models_2_experiments, found_models

def write_bar_plot(models_2_experiments, found_models):
    data = [] 
    y_label = METRIC_STR
    use_AUC = False
    if METRIC_STR == "AUC":
        use_AUC = True
        y_label = y_label + " (M)"
    min_val = np.inf
    for model in found_models:
        for run_num in range(len(list(models_2_experiments.values())[0])):
            value = np.mean([experiment[run_num] for experiment in models_2_experiments[model]])
            if use_AUC:
                value /= (10**6)
            event_dict = {"Model": model, 
                          y_label: value, 
                          "run_num": run_num,
                         }
            min_val = min(min_val, value)
            data.append(event_dict)
    fig = plt.figure()
    plt.xticks(rotation=90)
    if USE_TITLE:
        fig.suptitle("Overall Performance", fontsize=16)
    # Sort model order according to performance
    sorted_model_order = list(sorted(found_models, key = lambda model: np.mean(models_2_experiments[model])))
    plot = sns.barplot(x="Model", y=y_label, data=pd.DataFrame(data), 
        ci=68, order=sorted_model_order, palette=COLOR_PALETTE) # 68% confidence interval
    fig.set_size_inches(13*0.4, 7*0.4)
    plot.set_ylim(bottom=0.9*min_val) # Clip bottom of plot
    out_path = os.path.join(OUTPUT_DIR, METRIC_STR+"_Overall_Performance.png")
    plt.savefig(out_path, 
                dpi=700*(1/0.8), bbox_inches="tight")
    print("\n\nAgg Plot written to {}.".format(out_path))


def write_scatter_plot(models_2_experiments, found_models, snr_file, grad_file):
    print("Making scatter plot...")
    key_fn_pairs = [("snr", snr_file+".csv"), ("grad", grad_file+".csv"), ("perf", None)]
    model_to_ktriplet = defaultdict(list) # model to [snr, grad, perf]
    for key, fn in key_fn_pairs:
        if key == "perf": # third entry in triplet
            for model in MODEL_ORDER:
                if model == "SET": 
                # Set should not be included in scatter plot, 
                # since it has no order variance and cannot solve anything
                    continue
                if model not in found_models:
                    continue
                model_to_ktriplet[model].append(np.mean(models_2_experiments[model]))
            continue
        csv_path = os.path.join(RESULTS_DIR, fn)
        print("Incorporating {}".format(csv_path))
        with open(csv_path, "r") as file: # first 2 entries in triplet
            lines = file.readlines()
        for line in lines[1:]:
            model, metrics = line.split(";")
            metrics = [float(m) for m in metrics.split(",")]
            assert model in ALLOWED_MODELS, "{} not known model from {}".format(model, csv_path)
            model_to_ktriplet[model].append(np.mean(metrics))
            # AMRL-Avg has same snr as AVG, so just use AVG measurement
            if key == "snr" and model == "AVG":
                model_to_ktriplet["AMRL-Avg"].append(np.mean(metrics))
            # AMRL-Max has same snr as MAX, so just use MAX measurement
            if key == "snr" and model == "MAX":
                model_to_ktriplet["AMRL-Max"].append(np.mean(metrics))
    # Get data into proper format
    data = []
    x_label = "SNR"
    y_label = "Gradient"
    max_snr, max_grad = -100, -100
    min_snr, min_grad = 100, 100
    for model, triplet in model_to_ktriplet.items():
        if len(triplet) == 0:
            print("Warning no scatter plot data found for mode:", model)
            continue
        if len(triplet) != 3:
            print("Warning triplet for model", model, "is not length 3")
            continue
        snr, grad, perf = triplet
        if snr > max_snr:
            max_snr = snr
        if grad > max_grad:
            max_grad = grad
        if snr < min_snr:
            min_snr = snr
        if grad < min_grad:
            min_grad = grad
        event_dict = {"Model": model, 
                      x_label: snr, 
                      y_label: grad,
                      METRIC_STR: perf,
                     }
        data.append(event_dict)
    fig = plt.figure()
    if USE_TITLE:
        fig.suptitle("SNR-Gradient-Performance Scatter", fontsize=16)
    cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)
    plot = sns.scatterplot(x=x_label, y=y_label, data=pd.DataFrame(data), hue=METRIC_STR,
        size=METRIC_STR, palette=cmap)
    # add labels 
    for model, triplet in model_to_ktriplet.items():
        if len(triplet) != 3:
            continue
        snr, grad, perf = triplet
        if model == "AMRL-Avg": # Change text alignment, since it tends to overlap with sum on the plot
            alignment = "right"
        else:
            alignment = "left"
        plot.text(snr, grad, model, horizontalalignment=alignment, size=10, 
            color="black", weight="semibold")
    plot.set_xscale('log')
    plot.set_yscale('log')
    plot.set_xlim([min_snr/10,10*max_snr]) # snr
    plot.set_ylim([min_grad/10,10*max_grad]) # grad
    out_path = os.path.join(OUTPUT_DIR, METRIC_STR+"_Scatter.png")
    fig.set_size_inches(13*0.4, 7*0.4)
    plt.savefig(out_path, 
                dpi=700*(1/0.8), bbox_inches="tight")
    print("\n\nScatter Plot written to {}.".format(out_path))


def main():
    DEFAULT_EXPERIMENT_NAMES_STR = "TMaze Long, TMaze Long-Noise, TMaze Long-Short, TMaze Long-Short-Ordered, TMaze Long-Order-Variance, MC Long-Short, MC Long-Short-Noise, MC Long-Short-Ordered, MC Chicken"
    parser = ArgumentParser()
    parser.add_argument("--experiment_names", type=str, help="The names of the experiment csv files (created by visualize_runs.py) to aggregate.", 
                         default=DEFAULT_EXPERIMENT_NAMES_STR)
    parser.add_argument("--snr_file", type=str, help="The name of the SNR csv file (created by visualize_SNR.py --final) to aggregate.", 
                         default="SNR Final")
    parser.add_argument("--grad_file", type=str, help="The name of the gradient csv file (created by visualize_SNR.py --final) to aggregate.", 
                         default="Gradient Final")
    args = parser.parse_args()

    experiment_names = set([s.strip() for s in args.experiment_names.split(",") if s != ""]) # Split on commas; make sure unique
    print("\nAggregating metrics for model performance...")

    # Get dictionray of model to outcomes (5 in total for default AMRL) by experiment, represented as a list of equal-length lists.
    models_2_experiments, found_models = get_models_2_experiments(experiment_names)
    
    # Make bar plot of ocerall performance with condience interval
    write_bar_plot(models_2_experiments, found_models)
 
    # Make scatter plot for Gradient, SNR, Performance
    write_scatter_plot(models_2_experiments, found_models, args.snr_file, args.grad_file)



if __name__ == "__main__":
    main()
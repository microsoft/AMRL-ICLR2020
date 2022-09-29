'''
A script for visualizing the hidden states of LSTMs over episodes of training.
Requires the hidden states be saved as numpy files according to ray.rllib.examples.maze_runner.py.
Will also plot colors based on whether the inficator was up or down, as indicated in the name of the file.
'''

from argparse import ArgumentParser
from glob import glob
from abc import abstractmethod
from collections import defaultdict
import os
import numpy as np

import matplotlib as mpl
mpl.use('webagg')
mpl.rcParams['webagg.open_in_browser'] = False
mpl.rcParams['webagg.port'] = '8988'
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Dir specifying the save location of each gate neuron. (See t_maze.py for example.)
# Note that h and c are assumed to be stored as one vector in "state", and are not in this these dicts.
GATE_TO_DIR = {
    "f": "gate",
    "i": "i_gate",
    "o": "o_gate",
    "j": "j_proposal"
}
GATE_NEURONS = set(["f", "i", "o", "j"]) # Optional Gate neurons. (Technically j is not a gate.)

class TrajectoryProcessor:
    """ 
    Abstract class for converting np array of states of shape (episode_length, state_size) 
    into array of statistics of shape (epsiode_length, num_stats). num_stats will generally be 1.
    More stats can be returned (e.g. NoProc), but are currently not differentiated when plotted.
    """

    @abstractmethod
    def process(self, traj):
        """ Carry out the processing of the trajectory and return new trajectory of statistics """
        pass

    def __init__(self):
        self.plot_string = None

    def __call__(self, traj):
        new_traj = self.process(traj)
        assert len(traj) == len(new_traj), "{} != {}".format(len(traj), len(new_traj))
        return new_traj

class NoProc(TrajectoryProcessor):
    """ TrajectoryProcessor for returning all the activations of the states """ 

    def __init__(self):
        self.plot_string = "Activations"

    def process(self, traj):
        return traj

class VariancesProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the variances of states """ 

    def __init__(self):
        self.plot_string = "Variance"

    def process(self, traj):
        return np.var(traj, axis=1)

class L1Proc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the L1 norm of states """ 

    def __init__(self):
        self.plot_string =  "L1 Norm"

    def process(self, traj):
        return np.sum(np.abs(traj), axis=1)

class L2Proc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the L2 norm of states """ 

    def __init__(self):
        self.plot_string = "L2 Norm"

    def process(self, traj):
        return np.sqrt(np.sum(traj**2, axis=1))

class DistFromMeanProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the dostance to the mean of the trajectory """ 

    def __init__(self, L1=True):
        self.plot_string = "L1 to Mean" if L1 else "L2 to Mean"
        self.L1 = L1

    def process(self, traj):
        mean = np.mean(traj, axis=0)
        assert ((len(mean.shape) == 1) and (mean.shape == traj[0].shape)), (mean.shape, traj.shape)
        diff_to_mean = traj - mean
        return np.sum(np.abs(diff_to_mean), axis=1) if self.L1 else np.sqrt(np.sum(diff_to_mean**2, axis=1))

class LInfProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the infinity norm of states """ 

    def __init__(self):
        self.plot_string =  "LInf Norm"

    def process(self, traj):
        return np.max(np.abs(traj), axis=1)

class MeansProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the means of states """ 

    def __init__(self):
        self.plot_string = "Mean"

    def process(self, traj):
        return np.mean(traj, axis=1)

class MinProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the mins of states """ 

    def __init__(self):
        self.plot_string = "Min"

    def process(self, traj):
        return np.min(traj, axis=1)

class MaxProc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the maxs of states """ 

    def __init__(self):
        self.plot_string = "Max"

    def process(self, traj):
        return np.max(traj, axis=1)

class DiffL2Proc(TrajectoryProcessor):
    """ TrajectoryProcessor for caclulating the L2 norm of deltas of states """ 

    def __init__(self):
        self.plot_string = "L2 delta"

    def process(self, traj):
        new_traj = traj[1:,:] - traj[:-1,:] # subtract previous states
        new_traj = np.sqrt(np.sum(new_traj**2, axis=1))
        new_traj = np.insert(new_traj, 0, 0) # insert 0 delta at start to pad to same length
        return new_traj

def insert_initial_0_vector(traj):
    assert len(traj) > 0
    return np.vstack([np.expand_dims(np.zeros_like(traj[0]), axis=0), 
                      traj])

def get_numpy_files(parent_dir, directory):
    """ Read in numpy files from state, forget gate, or time directory, within parent_dir """
    assert directory in ["state", "time"]+list(GATE_TO_DIR.values()), directory
    file_dir = os.path.join(parent_dir, directory)
    assert os.path.isdir(file_dir), "{} does not exist. {} must contain {} dir".format(file_dir, parent_dir, directory)
    numpy_files = glob(os.path.join(file_dir, "*.npy"))
    print("Found", len(numpy_files), "{} episode files".format(directory))
    if not numpy_files:
        print("No {} files found".format(directory))
        exit()
    return numpy_files

def get_sample_indicies(num_total_samples, num_even_samples, num_endpoint_samples):
    """ 
    Figure out the sample indices given that we have a total number of samples avaialbe, and we 
    want a certain number of endpoint samples, and a certain number of evenly spaced samples.
    """
    sample_indicies = []
    num_endpoint_samples = min(num_total_samples//2, num_endpoint_samples) # Cap due to num samples total
    num_non_endpoint_sample = num_total_samples - 2*num_endpoint_samples
    num_even_samples = min(num_non_endpoint_sample, num_even_samples) # Cap due to num samples total
    # Add extra beginning endpoint samples
    for i in range(num_endpoint_samples):
        sample_indicies.append(i)
    if num_even_samples != 0:
        start_i = num_endpoint_samples
        end_i_exclusive = num_total_samples-num_endpoint_samples
        if num_even_samples == 1: # Needs special case
            sample_indicies.append(start_i + (end_i_exclusive-start_i)//2) # Append middle sample
        else:
            # Add evenly spaced sample indicies
            step_size = num_non_endpoint_sample/num_even_samples
            for float_i in np.arange(start_i, end_i_exclusive, step_size):
                sample_indicies.append(int(round(float_i)))
            # In case the numver of samples was not divided evenly, swap out the last sample for very last sample
            sample_indicies[-1] = num_total_samples-num_endpoint_samples-1
    # Add extra end endpoint samples
    for end_offset in reversed(range(num_endpoint_samples)):
        sample_indicies.append(num_total_samples-end_offset-1)
    return sample_indicies

def is_goal_up_from_filepath(fp):
    filename = fp.split("/")[-1]
    assert filename[0] in {"u", "d"}, "color only supported for tmaze goal position inidcated by these options."
    return filename[0] == "u"

def str_2_bool(s):
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError

def set_subplot_yaxis(subplot, range_tup):
    subplot.set_ylim([range_tup[0]-args.eps, range_tup[1]+args.eps])
    y_tick_locations = [range_tup[0], (range_tup[0]+range_tup[1])/2, range_tup[1]]
    subplot.set_yticks(y_tick_locations)
    y_tick_labels = [str(range_tup[0]), "", str(range_tup[1])]
    subplot.set_yticklabels(y_tick_labels)

def set_left_label(subplot, label):
    subplot.annotate(label, xy=(0, 0.5), xytext=(-90, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='small', ha='left', va='center')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="directory containing LSTM state dir (and forget gate dir)")
    parser.add_argument("--num_bins", type=int, help="the number of bins in histograms", default=15)
    parser.add_argument("--num_samples", type=int, help="the number of evenly spaced episode trajectories to plot", default=0)
    parser.add_argument("--alpha", type=float, help="The alpha for plot (0 is max transparency)", default=0.5)
    parser.add_argument("--eps", type=float, help="epsilon for plotting beyond known range of activation (to not cut off line)", default=0.1)
    parser.add_argument("--offset", type=int, help="offset added to non zero bins in histgrams to make brighter", default=5)
    parser.add_argument("--num_endpoint", type=int, help="the number of extra endpoint samples to plot, from both start and end", default=15)
    parser.add_argument("--zero_init", action='store_true', help="add additional 0 initial state", default=False)
    parser.add_argument("--prompt", action='store_true', help="prompt for which samples to use. Overrides other sample args.", default=False)
    parser.add_argument("--f", action='store_true', help="dispaly forget gate too", default=False)
    parser.add_argument("--i", action='store_true', help="dispaly input gate too", default=False)
    parser.add_argument("--o", action='store_true', help="dispaly output gate too", default=False)
    parser.add_argument("--j", action='store_true', help="dispaly proposal too", default=False)
    parser.add_argument("--color", type=str_2_bool, help="dispaly plot color by goal position", default=True)
    parser.add_argument("--traj", action='store_true', help="Whether to process and plot trajectories (or just histograms)", default=False)
    parser.add_argument("--exp", action='store_true', help="Whether to raise gates to the episode length for histograms", default=False)
    parser.add_argument("--bound_h", action='store_true', help="Whether to assume h is bounded (-1,1)", default=False)
    parser.add_argument("--AVG", action='store_true', help="If state was from AVG model, assume we should clip off the runnting time count stored in the last neuron", default=False)
    args = parser.parse_args()

    neurons = ["c", "h"] # Ordering of columns of plot (cells, hidden states, gates)
    if args.f:
        neurons.append("f")
    if args.i:
        neurons.append("i")
    if args.o:
        neurons.append("o")
    if args.j:
        neurons.append("j")
    gate_to_files = {}

    # Get numpy files
    numpy_state_files = get_numpy_files(args.dir, "state")
    for neuron_type in neurons:
        if neuron_type in GATE_NEURONS:
            numpy_gate_files = get_numpy_files(args.dir, GATE_TO_DIR[neuron_type])
            assert len(numpy_state_files) == len(numpy_gate_files), "{} != {}".format(len(numpy_state_files), len(numpy_gate_files))
            gate_to_files[neuron_type] = numpy_gate_files
    numpy_time_files = None
    if os.path.isdir(os.path.join(args.dir, "time")): # There are times saved for the episodes
        numpy_time_files = get_numpy_files(args.dir, "time")

    # Sort by creation time
    if numpy_time_files is not None:
        # Sort using time recorded in seperate file
        numpy_time_files.sort(key=lambda f: np.load(f))
        relative_filenames = [os.path.basename(f) for f in numpy_time_files]
        sorted_numpy_state_files = [os.path.join(args.dir,"state", f) for f in relative_filenames]
        assert set(sorted_numpy_state_files) == set(numpy_state_files), "Different episodes for state and time"
        numpy_state_files = sorted_numpy_state_files
        for neuron_type in neurons:
            if neuron_type in GATE_NEURONS:
                sorted_numpy_gate_files = [os.path.join(args.dir, GATE_TO_DIR[neuron_type], f) for f in relative_filenames]
                assert set(sorted_numpy_gate_files) == set(gate_to_files[neuron_type]), "Different episodes for gate and time"
                gate_to_files[neuron_type] = sorted_numpy_gate_files
    else:
        # Sort by ctime
        numpy_state_files.sort(key=os.path.getctime)
        for neuron_type in neurons:
            if neuron_type in GATE_NEURONS:
                gate_to_files[neuron_type].sort(key=lambda f: (os.path.getctime, f))
                relative_gate_filenames = [os.path.basename(f) for f in gate_to_files[neuron_type]]
                relative_state_filenames = [os.path.basename(f) for f in numpy_state_files]
                assert set(relative_state_filenames) == set(relative_gate_filenames), "Different episodes for state and gate"
                if relative_state_filenames != relative_gate_filenames:
                    print("Warning: Sorting the state files and gate files produced different results.")
                    print("This may just be 2 episodes that wrote to disk at the same time. \nDefaulting to state ctime sort.")
                    gate_to_files[neuron_type] = [os.path.join(args.dir, GATE_TO_DIR[neuron_type], f) for f in relative_state_filenames]

    # Find episodes we want to plot and create title
    if args.prompt:
        print("Please list the episode numbers you would like to use:")
        sample_indicies = tuple(map(int, input().split(",")))
        title_str = "State Trajectories \nepisodes: {} of {}".format(sample_indicies, len(numpy_state_files))
        plot_all_hists = True # Whether to plot all histograms or just 1st and last
    else:
        sample_indicies = get_sample_indicies(len(numpy_state_files), args.num_samples, args.num_endpoint)
        title_str = "State Trajectories \n({} even samples, {} endpoint samples, of {})".format(args.num_samples, 
                                                                                               args.num_endpoint,
                                                                                               len(numpy_state_files))
        plot_all_hists = False # Too many too plot all
    print("Using episodes:", sample_indicies)
    # Define plots
    stats_processors = [L1Proc(), L2Proc(), 
                        LInfProc(), MeansProc(), MinProc(), MaxProc(),
                        VariancesProc(), DiffL2Proc(), DistFromMeanProc(L1=False), DistFromMeanProc(L1=True), NoProc()] # Ordering of rows
    num_cols = len(neurons)
    num_hists = len(sample_indicies) if plot_all_hists else 2
    num_traj_plots = len(stats_processors) if args.traj else 0
    sub_plot_grid_shape = (num_traj_plots+num_hists+1, num_cols) #+1 for mean by episode
    neuron_to_range = {"c": None, "h": (-1,1) if args.bound_h else None, "f": (0,1), "i": (0,1), "o": (0,1), "j": (-1,1)}
    # set titles
    neuron_to_title = {
        "c": "Cell", 
        "h": "H", 
        "f": "Forget Gate", 
        "i": "Input Gate",
        "o": "Output Gate",
        "j": "Proposal Gate"
    }
    fig, subplots = plt.subplots(nrows=sub_plot_grid_shape[0], ncols=sub_plot_grid_shape[1])
    fig.suptitle(title_str, fontsize=16)
    for col, neuron_type in enumerate(neurons):
        subplots[0][col].set_title(neuron_to_title[neuron_type])  
    # set cols' x-axes
    for i in range(num_cols):
        subplots[-1][i].set_xlabel("Step")

    # iterate over relevant data to plot
    neuron_to_traj_mean_by_episode = defaultdict(list)
    for episode_plot_num, i in enumerate(sample_indicies):
        frac_done = i/len(numpy_state_files)
        # Load and split state trajectories
        episode_state_trajectory = np.load(numpy_state_files[i])
        # Confirm that episode_state_trajectory has shape of length 3
        # This should be: (episode_length, 2, lstm_size)
        assert len(episode_state_trajectory.shape) == 3, len(episode_state_trajectory.shape)
        state_ep_len, dim2, lstm_size = episode_state_trajectory.shape
        assert dim2 == 2 # There should be 1 hidden state and 1 cell state
        # assume 0 init if desired
        if args.zero_init:
            episode_state_trajectory = insert_initial_0_vector(episode_state_trajectory)
        # Split into cells and hidden states
        cs = episode_state_trajectory[:,0,:] # cells
        hs = episode_state_trajectory[:,1,:] # hidden states
        if args.AVG:
            cs = cs[:, :-1] # Get rid off running time count
            hs = hs[:, :-1]
        neuron_to_traj = {"c": cs, "h": hs}
        neuron_to_traj_mean_by_episode["c"].append(np.mean(cs))
        neuron_to_traj_mean_by_episode["h"].append(np.mean(hs))
        for neuron_type in neurons:
            if neuron_type in GATE_NEURONS:
                # Load forget gate trajectory
                gs = np.load(gate_to_files[neuron_type][i]) # gates
                neuron_to_traj[neuron_type] = gs
                neuron_to_traj_mean_by_episode[neuron_type].append(np.mean(gs))
                # Confirm that gs has shape of length 2
                # This should be: (episode_length, lstm_size)
                assert len(gs.shape) == 2, len(gs.shape)
                assert lstm_size == gs.shape[-1], "lstm sizes {} and {} should be equal".format(lstm_size, gs.shape[-1])
                if abs(gs.shape[0] - state_ep_len) > 1:
                    # Note, these lengths will likely differ by 1 since lstms have an initial state but forget gates do
                    # not have an actiavtion until the first step has been taken
                    print("Warning, episode lengths for state and gate differ by > 1: {} and {}".format(state_ep_len, gs.shape[0]))
        # Plot histograms of final activations over time
        is_start, is_end = (i == sample_indicies[0]), (i == sample_indicies[-1])
        if plot_all_hists or is_start or is_end:
            if plot_all_hists:
                row = episode_plot_num
            else:
                row = 0 if is_start else 1
            set_left_label(subplots[row][0], "Histogram \n(episode {})".format(i))
            for col, neuron_type in enumerate(neurons):
                subplot = subplots[row][col]
                traj, act_range = neuron_to_traj[neuron_type], neuron_to_range[neuron_type]
                if neuron_type not in GATE_NEURONS:
                    traj = traj[1:] # The first is all 0s, means all neurons in same bucket, which throws off max for color map
                if args.exp and neuron_type == "f":
                    traj = np.power(traj, len(traj))
                    subplot.set_ylabel(neuron_type+" ^T", fontsize=8)
                    subplot.yaxis.set_label_coords(0, 0.5)
                if act_range is None:
                    act_range = traj.min(), traj.max() # min, max over whole trajectory
                hist_func = lambda x: np.histogram(x, bins=args.num_bins, range=act_range)[0]
                hist = [hist_func(gates) for gates in traj]
                hist = np.flip(np.transpose(np.array(hist, dtype=np.float)), axis=0)
                hist = np.where(hist==0, 0, hist+args.offset)
                hist = np.pad(hist, pad_width=1, mode='constant', constant_values=0) # so that white on border shows up
                max_hist_count_for_norm = len(traj[0]) # absolute normalization for histogram (white = all nuerons in bin)
                max_hist_count_for_norm += args.offset
                subplot.imshow(hist, vmin=0, vmax=max_hist_count_for_norm, aspect="auto")
                lim = subplot.get_ylim()
                subplot.set_yticks(lim)
                subplot.set_yticklabels([np.round(x, decimals=1) for x in act_range])
                subplot.set_xticks([])
                if row == num_hists-1:
                    subplot.set_xlabel("episode percent", fontsize=8)
                    subplot.xaxis.set_label_coords(0.5, 0)
        # Plot all processors
        if args.traj:
            for row, stats_proc in enumerate(stats_processors):
                row = row + num_hists + 1 # Leave first rows for histograms and means
                for col, neuron_type in enumerate(neurons):
                    traj, act_range = neuron_to_traj[neuron_type], neuron_to_range[neuron_type]
                    subplot = subplots[row][col]
                    if row < sub_plot_grid_shape[0]-1: # Turn off tick for all except last row
                        subplot.set_xticklabels([])
                    if col == 0: # Add row annotation
                        set_left_label(subplot, stats_proc.plot_string)
                    if act_range is not None and act_range[0] >= 0 and type(stats_proc) is LInfProc:
                        # Since gates are > 0, this is same as max. Don't plot.
                        subplot.set_xticks([])
                        subplot.set_yticks([])
                        continue
                    stats_trajectory = stats_proc(traj) # Do actual trajectory processing
                    if args.color:
                        # Color by fraction done and goal position
                        goal_up = is_goal_up_from_filepath(numpy_state_files[i])
                        if neuron_type in GATE_NEURONS:
                            assert goal_up == is_goal_up_from_filepath(gate_to_files[neuron_type][i]), \
                            "Right now coloring by goal position is only supported if it is the same for state and gate. See earlier warning."
                        color = (0,frac_done,0) if goal_up else (0,0,frac_done) # blue for down
                    else:
                        # Color by fraction done
                        color = (0,frac_done,0)
                    # Plot
                    subplot.plot(stats_trajectory, color=color, alpha=args.alpha) # Do actual plotting
                    # Set range manually for plots bounded by known range
                    if act_range is not None and type(stats_proc) in [LInfProc, MeansProc, MinProc, MaxProc, VariancesProc, NoProc]:
                        set_subplot_yaxis(subplot, act_range)
    
    # Plot mean by episode for each col
    set_left_label(subplots[num_hists][0], "Trajectory\nMean")
    for col, neuron_type in enumerate(neurons):
        subplot = subplots[num_hists][col]
        subplot.plot(sample_indicies, neuron_to_traj_mean_by_episode[neuron_type])
        subplot.set_xlabel("episode num", fontsize=8)
        subplot.xaxis.set_label_coords(0.5, 0)
        act_range = neuron_to_range[neuron_type]
    
    # Show all plots
    plt.show()



'''
A python script for a docker image on the cluster to call to run t_maze or mine_maze with options below.
'''

import os
from subprocess import call
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--link", action="store_true", default=False, help="Set up symlinks for rllib")
parser.add_argument("--redis_address", type=str, default=None, help="Redis address from start script")
parser.add_argument("--workers_sync_count", type=int, default=None, help="Wait for given number of workers in ray cluster (includes head)")
args, unkown = parser.parse_known_args()

if args.link:
    print("\n\nLINKING...\n\n")
    # Create symlinks to work out of cloned dir, not built ray
    # Note: This works 99% of the time; if it doesn't, then just try again!
    setup_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    settup_file = os.path.join(setup_dir, "setup-dev.py")
    settup_command = "python3 " + settup_file + " --yes"
    call(settup_command, shell=True)
else:
    print("\n\nRUNNING...\n\n")
    # Run experiment
    # e.g. (tmaze, minemaze, or chicken) --grid --num_runs=5
    call("python3 maze_runner.py chicken --dnc_first --folder --clust --grid --num_runs=5 --redis_address " + args.redis_address + " --workers_sync_count " + str(args.workers_sync_count), shell=True)

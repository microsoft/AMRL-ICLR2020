# A script to recreate experiments from AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html). Stores results in ./AMRL_results
# Note: "set" model has been removed from experiments in order to simplify implementation.
# Note: Due to compute budget, this script has not been tested and should be considered experimental.
# Note: This was not created to work with a ray cluster, and may be slow without one.

cd "`dirname \"$0\"`" # cd into RL_Memory directory

# Run the experiments, grouped by env
cd python/ray/rllib/examples
python maze_runner.py --grid --yaml_dir=tmaze     tmaze
python maze_runner.py --grid --yaml_dir=mine_maze mine_maze
python maze_runner.py --grid --yaml_dir=chicken   chicken

# Plot the Learning Curves and Calculate Metrics (best over all lrs), for each experiment.
cd ..
# tmaze
python visualize_runs.py T-L   --no_show --out_name="T-L_plot"   --best --title
python visualize_runs.py T-LN  --no_show --out_name="T-LN_plot"  --best --title
python visualize_runs.py T-LO  --no_show --out_name="T-LO_plot"  --best --title
python visualize_runs.py T-LS  --no_show --out_name="T-LS_plot"  --best --title
python visualize_runs.py T-LSO --no_show --out_name="T-LSO_plot" --best --title
# mine_maze
python visualize_runs.py MC-LS  --no_show --out_name="MC-LS_plot"  --best --title
python visualize_runs.py MC-LSN --no_show --out_name="MC-LSN_plot" --best --title
python visualize_runs.py MC-LSO --no_show --out_name="MC-LSO_plot" --best --title
# mine_chicken
python visualize_runs.py MC_Chicken --no_show --out_name="MC_Chicken_plot" --best --title

# Calculate and plot SNR, for each model
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --no_show --out_name=snr_plot # Line plot
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --final --no_show --out_name="snr_final" # Bar plot and calculation

# Calculate and plot Gradient Decay, for each model
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --grad --signal --lower_ylim=10e-15 --upper_ylim=10 --no_show --out_name=grad_plot # Line plot  
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --final --grad --signal --no_show --out_name="grad_final" # Bar plot and calculation

# Combine Metrics, SNR, and Gradient Decay calculations above into new plot.
python aggregate_exper.py --snr_file="snr_final" --grad_file="grad_final" --experiment_names="T-L_plot, T-LN_plot, T-LO_plot, T-LS_plot, T-LSO_plot, MC-LS_plot, MC-LSN_plot, MC-LSO_plot, MC_Chicken_plot"

# Results now in: ./AMRL_results

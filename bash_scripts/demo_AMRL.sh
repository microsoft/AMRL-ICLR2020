# A script to demo experiments from AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html). Stores results in ./AMRL_results
# This script uses AMRL-Max and LSTM on two TMaze environments to show resiliency of AMRL-Max to noise.
# It also creates a bar plot for all models (other than "set") in the paper of gradient decay and SNR.
# Note: "set" model is left out of these plots since "set" is not implemented in the new architecture interface.
#       (It is fairly uselss anyway, since it cannot remember ordered events.)

cd "`dirname \"$0\"`" # cd into RL_Memory directory

# Run the RL Agents (Note: if re-run. maze_runner removes failed runs and restarts, instead of having rllib attempt to resume)
cd python/ray/rllib/examples
python maze_runner.py --grid --yaml_dir=DEMO tmaze

# Plot the Learning Curves and Calculate Metrics (best over all lrs)
cd ..
python visualize_runs.py T-L --no_show --out_name="demo_plot" --best  --lower_ylim=-.3 --title        # T-L results
python visualize_runs.py T-LN --no_show --out_name="demo_plot_noise" --best  --lower_ylim=-.3 --title # T-LN results

# Calculate and plot SNR
# Note: The yaml directory must contain yamls at the top-level. These yaml files only are used to define the model architecture.
#       It therefore does not matter whether you use "DEMO/TMaze Long/models" or "DEMO/TMaze Long Noise/models", since the environment is ignored.
python visualize_SNR.py --yaml_dir="DEMO/TMaze Long/models" --no_show --out_name=demo_snr # Line plot
python visualize_SNR.py --yaml_dir="DEMO/TMaze Long/models" --final --no_show --out_name="demo_snr_final" # Bar plot and calculation
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --final --no_show --out_name=snr_final # Bar plot for all models! Why not take a look?

# Calculate and plot Gradient Decay
python visualize_SNR.py --yaml_dir="DEMO/TMaze Long/models" --grad --signal --lower_ylim=10e-15 --upper_ylim=10 --no_show --out_name=demo_grad # Line plot  
python visualize_SNR.py --yaml_dir="DEMO/TMaze Long/models" --final --grad --signal --no_show --out_name="demo_grad_final" # Bar plot and calculation
python visualize_SNR.py --yaml_dir="chicken/MC Chicken/models" --final --grad --signal --no_show --out_name=grad_final # Bar plot for all models! Why not take a look?

# Combine Metrics, SNR, and Gradient Decay calculations above into new plot. (arguments same as final out files above)
python aggregate_exper.py --experiment_names="demo_plot, demo_plot_noise" --snr_file="demo_snr_final" --grad_file="demo_grad_final"

# Results now in: ./AMRL_results

# To recreate the figures
wihtout running any simulations, or to play around with our data, follow these steps:

1. Clone this repository
2. Set up a python environment with the dependencies listed in `requirements.txt`, e.g.:

		python -m venv .env
		pip install -r requirements.txt

3. Download the abridged data from https://doi.org/10.6084/m9.figshare.26796061 and unzip it as a `results/` subdirectory
4. Run all jupyter notebooks to generate the main and supplementary figures.



# To replicate the published results without data,
by running new simulations, you will need a Linux or similar system. Windows users are recommended to use WSL. Simulation and analysis are possible in any system, but will require a little more footwork to replicate the work the bash scripts (steps 3-4 below) do. Luckily, the scripts are very simple (go look), and are easily replaced with some copy-paste drudgery. If that's not for you, follow these steps:

1. Clone this repository

2. Set up a python environment with the dependencies listed in `requirements.txt`, e.g.:

		python -m venv .env
		pip install -r requirements.txt

3. Run simulations with a specified runtime seed, which should be a positive integer:

		bash grid-run.sh <runseed>
	Note:
    1. This runs through all combinations of p_inh and r_inh as defined in the parameter files in `params/`. Note that each parameter file itself defines 10 networks to be run in parallel (`N_nets=10`), generated with a consistent random seed for structure (`rng=0`).
    2. To recreate the published results exactly (minus possible floating point inaccuracies etc.), call `grid_run.sh` (and subsequent scripts) with runseeds 0, 100, 200, 300, and 400 in five separate calls.
    3. Every run of a single parameter setting generates ~1.3G of raw data; i.e., invoking `grid-run.sh` once requires on the order of ~35G of disk space.
    4. Caution: There is no overwrite protection. If you are not careful with the naming scheme, the contents of existing subdirectories in `results/` may be overwritten.

4. Run analysis scripts to munge the raw data into manageable size, again using the runseed as an integer argument:

		bash grid-spikestats.sh <runseed>
		bash grid-wstats.sh <runseed>

	Notes (i) and (ii) above apply here, too.

5. Run postprocessing script for further data munging:

		python grid_postprocessing.py <runseed[s]>
 
	Unlike the bash scripts, this can be called with multiple runseed arguments (e.g., `python grid_postprocessing.py 0 100 200 300 400`), but note that it relies on the outputs of step 4 above.

6. Run all jupyter notebooks to generate the main and supplementary figures.

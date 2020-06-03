# Instructions

This code is a supplement for "A generic adaptive restart scheme with
applications to saddle point algorithms".

## One-time setup

*All instructions assume that your current working directory is the base of this
repository.*

The supplement contains a mix of Julia and Python code, so both environments
must be set up.

For Julia, install Julia 1.4.2 from https://julialang.org/downloads/ (other versions of
Julia may work also) and make sure `julia` is available in your `PATH`.

Instantiate the Julia packages:

```
$ julia --project=. -e "import Pkg; Pkg.instantiate()"
```

For Python, install pipenv:

```
$ sudo apt install python3-pip  # Or the appropriate command for your system.
$ pip3 install --user pipenv
# Modify .bashrc to add $HOME/.local/bin to PATH.
```

Then in a shell with the new `PATH` loaded:

```
$ pipenv install
$ pipenv run python3 -c "import odl; odl.test()"
```

## pipenv setup

This was used to generate the pipenv environment. It shouldn't be necessary to
re-run this.

```
$ pipenv install -e git+https://github.com/odlgroup/odl@2320e398bcbb96cdf548d0f08b177a54d8ab7e7e#egg=odl[testing]
$ pipenv install cvxopt cvxpy h5py pandas
```

## Running the primal-dual hybrid gradient (PDHG) code

To run the matrix games instances and generate corresponding plots:

```
$ pipenv run python3 -m pdhg.pdhg_matrix_games $HOME/pdhg_results
$ julia --project=. pdhg/plot_matrix_games_results.jl $HOME/pdhg_results
```

The output will be at `$HOME/pdhg_results`. The experiments may take a couple
hours to run.

To reproduce our calibration of the primal/dual step sizes, run the following:

```
$ pipenv run python3 -m pdhg.lp_calibrate_tau_sigma_ratio pdhg/qap15.hdf5
$ pipenv run python3 -m pdhg.lp_calibrate_tau_sigma_ratio pdhg/nug08-3rd.hdf5
```

For each instance this will print a table of residuals after 1000 iterations of
PDHG without restarts. The ratio is chosen to be the one with the smallest
value.

To run the quadratic assignment problem (linear programming) instances, run the
following:

```
$ pipenv run python3 -m pdhg.lp_experiment pdhg/qap15.hdf5 $HOME/pdhg_results/qap15 0.0001 300000
$ julia --project=. pdhg/plot_lp_results.jl $HOME/pdhg_results/qap15
```

```
$ pipenv run python3 -m pdhg.lp_experiment pdhg/nug08-3rd.hdf5 $HOME/pdhg_results/nug08-3rd 0.01 10000
$ julia --project=. pdhg/plot_lp_results.jl $HOME/pdhg_results/nug08-3rd
```

These will output detailed logs and plots at `$HOME/pdhg_results/qap15` and
`$HOME/pdhg_results/nug08-3rd` respectively. The experiments may take a couple
hours to run.

## Running the accelerated gradient descent (AGD) code

Run

```
$ sh agd/download_lib_svm.sh
```

to collect the dataset. This script creates directories `agd/data` and
`agd/plots` and downloads data to the `data` directory.

Test the code is working by running:

```
$ julia --project=. agd/tests/test.jl
```

To reproduce the AGD results in the paper:

```
$ julia --project=. agd/lasso_example.jl E2006.train 1.0
$ julia --project=. agd/logistic_example.jl rcv1_train.binary 0.1
$ julia --project=. agd/logistic_example.jl duke.tr 1e-2
$ julia --project=. agd/hard_example.jl 500 1e-4 1e-4
```

Output will be saved in the `agd/plots` folder. Note these results may take
several hours to run (primarily due to hyperparameter searches).

## Linear programming instances

The `qap15` and `nug08-3rd` instances are included in the repository in a
bespoke HDF5 format. The following steps document how to regenerate these files
from the original sources.

The `nug08-3rd` instance is available in "compressed MPS" format
[here](http://plato.asu.edu/ftp/lptestset/nug/nug08-3rd.gz). See
[these notes](http://plato.asu.edu/ftp/lptestset/nug/00README) on decompressing
the file into MPS using a special utility.

The generator for the `qap15` instance is available
[here](http://www.netlib.org/lp/generators/qap/). The generator outputs an MPS
file.

Once in MPS format, use the following commands to generate the HDF5 files:

```
$ julia --project=. pdhg/dump_to_hdf5.jl /path/to/qap15.mps pdhg/qap15.hdf5
$ julia --project=. pdhg/dump_to_hdf5.jl /path/to/nug08-3rd.mps pdhg/nug08-3rd.hdf5
```

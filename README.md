# Group 9 AI Safety Project

Please refer to the report for more details.

# Installation
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

After you've installed poetry, you can install partx by running the following command in the root of the project: 

```
poetry install
```
# Part-X Codebase

Part-X Codebase taken from [https://github.com/cpslab-asu/part-x/tree/main](https://github.com/cpslab-asu/part-x/tree/main)

```
@article{pedrielli2023part,
    title={Part-x: A family of stochastic algorithms for search-based test generation with probabilistic guarantees},
    author={Pedrielli, Giulia and Khandait, Tanmay and Cao, Yumeng and Thibeault, Quinn and Huang, Hao and Castillo-Effen, Mauricio and Fainekos, Georgios},
    journal={IEEE Transactions on Automation Science and Engineering},
    year={2023},
    publisher={IEEE}
}
```
# Running the codes

All the plots and the csv files have been generated. Data could not be uploaded since the the toal upload size exceeded 5GB.

## To train the controller:

```
cd demos/safety_partx
poetry run python generate_model.py
```

## To run UR Benchmarks - 3d Case

```
cd demos/safety_partx
poetry run python run_UR_3d.py
```

## To run UR Benchmarks - 7d Case

```
cd demos/safety_partx
poetry run python run_UR_7d.py
```

## To run Part-X Benchmarks - 3d Case

```
cd demos/safety_partx
poetry run python run_ptx_benchmark_3d.py
```

## To run Part-X Benchmarks - 7d Case

```
cd demos/safety_partx
poetry run python run_ptx_benchmark_7d.py
```

## To generate plots for the 3d case:

```
cd demos/safety_partx
poetry run python model_3d_plots.py
```

## To generate plots for the 7d case:

```
cd demos/safety_partx
poetry run python model_7d_plots.py
```

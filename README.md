
# Missing Data in Epidemiological Forecasting

## About This Repository

This repository contains the codebase for a bachelor thesis project comparing various missing data imputation techniques under different assumptions of missing data patterns.

The `main` branch contains all the code developed before the thesis deadline, while other branches include sections designated as "future work."

## Thesis Project Abstract

Poor-quality data makes unbiased and reliable data-driven decision-making impossi-
ble in the field of epidemiological forecasting. While there are many forms of data
distortion, the presence of missing values in datasets is one of the most well-studied
and crucial areas to address in the domain. Yet, published studies often only focus on
specific available time series and test one pattern of omitted records. This results in
the absence of any universal recommendations formulated to address such an issue.
In our study, we evaluate nine techniques for handling missing data: rolling mean,
LOCF, NOCB, KNN, linear and quadratic interpolation, ARIMA, Kalman filter, and
page matrix estimation. We test the methods on three series representing daily records
of new COVID-19 cases from three different countries: Italy, the United Kingdom,
and Israel. We conduct experiments under three different missing value patterns, cor-
responding to the varying nature of missingness, and at six different percentages of
missing values. We conclude that KNN, linear interpolation, and Kalman filter meth-
ods are good starting point recommendations for the majority of assessed scenarios.

## Data Used

The main dataset for this project is the WHO COVID-19 cases dataset, available at:

[World Health Organization COVID-19 Dashboard](https://data.who.int/dashboards/covid19/cases)

## Directory Structure

- **[data](data/)**: Contains the CSV file with the WHO dataset.
- **[results](results/)**: Contains the results of simulation runs, including mean, median, standard deviation, maximum, and minimum values of MAPE scores for each country and missing data percentage. Separate folders are provided for each missing data pattern.
- **[visualizations](visualizations/)**: Contains all the plots used in the thesis.
- **[ml_imputer_class.py](ml_imputer_class.py)**: Defines a class with static methods for the different imputation techniques tested (KNN, Kalman filter, matrix factorization, ARIMA).
- **[visualization_utils.py](visualization_utils.py)**: Contains functions for generating visualizations.
- **[simulation_utils.py](simulation_utils.py)**: Contains helper functions for running the experimental simulations.
- **[simulation.py](simulation.py)**: The main module for running experiments. Usage:

```sh
python simulation.py --nature <NATURE> --n_simulations <N>
```
  - `NATURE` argument can be one of `MCAR`, `MAR`, or `MNAR`.
  - `N` argument specifies the number of simulations (default: 500).

- **[results_analysis.ipynb](results_analysis.ipynb)**: Jupyter notebook containing final metrics tables and visualizations.

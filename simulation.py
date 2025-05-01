import pandas as pd
import numpy as np
import argparse


from ml_imputer_class import MLImputer
from simulation_utils import *

import warnings
warnings.filterwarnings("ignore")

who = pd.read_csv("data/WHO-COVID-19-global-daily-data.csv")
who['Date_reported'] = pd.to_datetime(who['Date_reported'])

countries = [
    {'country': 'Italy',
     'start_date': '2020-02-23', 
     'end_date' : '2023-10-30'},

    {'country': 'United Kingdom of Great Britain and Northern Ireland',
     'start_date': '2020-03-10', 
     'end_date' : '2023-10-22'},

    {'country': 'Israel',
     'start_date': '2020-03-25', 
     'end_date' : '2023-10-29'}
]

methods = {'Rolling Mean': lambda arr: pd.Series(arr).fillna(pd.Series(arr).rolling(3,min_periods=1).mean()).fillna(np.nanmean(arr)),
           'LOCF': lambda arr: pd.Series(arr).fillna(method='ffill').fillna(method='bfill').to_numpy(),
           'NOCB': lambda arr: pd.Series(arr).fillna(method='bfill').fillna(method='ffill').to_numpy(),
           'Linear Interpolation': lambda arr: pd.Series(arr).interpolate(method='linear', limit_direction='both').to_numpy(),
           'Quadratic interpolation': lambda arr: pd.Series(arr).interpolate(method='quadratic', limit_direction='both').fillna(np.nanmean(arr)).to_numpy(),
           'KNN': lambda arr: MLImputer.knn_impute(arr),
           'Kalman Filter': lambda arr: MLImputer.kalman_impute(arr),
           'Matrix Estimation': lambda arr: MLImputer.matrix_estimation_impute(arr),
           'ARIMA': lambda arr: MLImputer.arima_impute(arr)}

if __name__=="__main__":
    # Saved results are with old methods names
    countries_data = {}
    for country in countries:
        countries_data[country['country']] = get_country_series(who, country['country'], country['start_date'], country['end_date'])
    
    for key in countries_data:
        parser = argparse.ArgumentParser()
        parser.add_argument("--nature", type=str,
                        help="Missing data nature of interest")
        parser.add_argument("--n_simulations", "-n", type=int,
                        help="number of simulations to run")

        args = parser.parse_args()
        run_simulations(key, countries_data[key], methods=methods, nature=args.nature, n_simulations=args.n_simulations)

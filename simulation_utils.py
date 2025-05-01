import pandas as pd
import numpy as np

from ml_imputer_class import MLImputer


def add_noise(series, noise_perc = 0.1, nature = "MCAR"):
  if nature == "MCAR":
    series = series.astype(np.float64)
    n = int(np.ceil(len(series) * noise_perc))
    
    idx_replace = np.random.randint(0, len(series), n)

    nan_series = series.copy()
    nan_series[idx_replace] = np.nan
  elif nature == "MAR":
    df_copy = series.copy()
    n = int(np.ceil(len(df_copy) * noise_perc))
    
    days = df_copy['Date_reported'].dt.dayofweek
    probs = np.ones(len(df_copy))
    
    probs[(days == 5) | (days == 6)] *= 2.0
    
    probs = probs / probs.sum()

    idx_replace = np.random.choice(df_copy.index, size=n, replace=False, p=probs)

    nan_series = df_copy['New_cases'].copy()
    nan_series.loc[idx_replace] = np.nan
    nan_series = np.array(nan_series.values)
  else:
    n = int(np.ceil(len(series) * noise_perc))

    diff = np.abs(np.diff(series, prepend=series[0]))

    scaled_diff = (diff - np.nanmin(diff)) / (np.nanmax(diff) - np.nanmin(diff) + 1e-8)

    probs = scaled_diff ** 1.5
    probs /= probs.sum()

    idx_replace = np.random.choice(len(series), size=n, replace=False, p=probs)

    nan_series = series.copy()
    nan_series[idx_replace] = np.nan


  return nan_series


def mape(gt, estimate):
    gt = np.array(gt)
    estimate = np.array(estimate)

    diff_mask = (gt != estimate) & (gt != 0)


    return np.mean(np.abs((gt[diff_mask] - estimate[diff_mask]) / gt[diff_mask]))


def get_country_series(who, country, start_date=None, end_date=None):
  country_data = who[who['Country'] == country]
  if not start_date:
    country_data = country_data.loc[(country_data['Date_reported'] <= end_date)]
  elif not end_date:
    country_data = country_data.loc[(country_data['Date_reported'] >= start_date)]
  else:
    country_data = country_data.loc[(country_data['Date_reported'] >= start_date) & (country_data['Date_reported'] <= end_date)]
  country_data = country_data[['Date_reported', 'New_cases', 'New_deaths']]
  country_data['New_deaths'] = country_data['New_deaths'].fillna(0)

  return country_data


def create_summary_table(missing, results):
  results_t = []
  for method in results:
    results_t.append({'missing': missing,
                      'method': method,
                      'mean': np.nanmean(results[method]),
                      'median': np.nanmedian(results[method]),
                      'std': np.nanstd(results[method]),
                      'min': np.nanmin(results[method]),
                      'max': np.nanmax(results[method]),})

  return pd.DataFrame(results_t)


def run_simulations(country, df, methods, nature = "MCAR", n_simulations=500, missing_percentages=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25], methods_subspace=''):
    print(f'Running simulation on {country} data')
    nc = np.array(df['New_cases'])

    results_t_nc_mape = pd.DataFrame(columns=['missing', 'method', 'mean', 'std', 'min', 'max'])
    results_nc_mape = {x: [] for x in methods.keys()}

    for perc_miss in missing_percentages:
        print(f'running simulation for {perc_miss*100}% of missing values')
        for i in range(n_simulations):
            if nature == "MAR":
                curr_series_nc = add_noise(df, noise_perc=perc_miss, nature="MAR")
            else:
                curr_series_nc = add_noise(nc, noise_perc=perc_miss, nature=nature)
            for method in methods:
                imputed = methods[method](curr_series_nc)
                results_nc_mape[method].append(mape(nc, imputed))


        summary_mape = create_summary_table(perc_miss, results_nc_mape)
        results_t_nc_mape = pd.concat([results_t_nc_mape, summary_mape], ignore_index=True)

        results_nc_mape = {x: [] for x in methods.keys()}

    results_t_nc_mape.to_csv(f"results/covid_{nature}/mape/{country}_{n_simulations}{methods_subspace}_cases_mape_results.csv", index=False)

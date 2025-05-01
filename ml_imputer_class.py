import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from pykalman import KalmanFilter
from fancyimpute import SoftImpute
from pmdarima import auto_arima
from numpy import ma 


class MLImputer:
    """
    Collection of static methods for univariate time series imputation.
    """

    @staticmethod
    def knn_impute(
        arr: np.ndarray,
        n_neighbors: int = 3,
    ) -> np.ndarray:
        """
        Impute missing values using K-Nearest Neighbors.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        arr_df = pd.DataFrame()
        arr_df['records'] = arr
        impute_result = imputer.fit_transform(arr_df['records'].reset_index())
        filled = [i[1] for i in impute_result]
        return filled

    @staticmethod
    def kalman_impute(
        arr: np.ndarray,
    ) -> np.ndarray:
        """
        Impute missing values using a univariate Kalman Filter.
        """
        measurements = ma.masked_invalid(arr)
        kf = KalmanFilter()
        
        kf = kf.em(measurements, n_iter=10)
        smoothed_state_means, _ = kf.smooth(measurements)
        
        imputed = np.where(np.isnan(arr), smoothed_state_means[:, 0], arr)
        
        return imputed

    @staticmethod
    def matrix_estimation_impute(
        series: np.ndarray,
        window_size: int = 20,
        rank: int = 5,
        max_iters: int = 500
    ) -> np.ndarray:
        """
        Impute missing values via low-rank matrix estimation (SoftImpute).
        """
        n = len(series)
        n_trimmed = (n // window_size) * window_size
        trimmed_series = series[:n_trimmed]

        n_cols = n_trimmed // window_size
        page_matrix = trimmed_series.reshape((n_cols, window_size)).T

        soft_impute = SoftImpute(max_rank=rank, max_iters=max_iters, verbose=False)
        completed_matrix = soft_impute.fit_transform(page_matrix)

        if np.isnan(completed_matrix).all():
            print("Warning: SoftImpute produced only NaNs! Falling back to linear interpolation + mean filling.")
            imputed_trimmed = pd.Series(trimmed_series).interpolate(method='linear', limit_direction='both').fillna(np.nanmean(trimmed_series)).to_numpy()
        else:
            imputed_trimmed = completed_matrix.T.flatten()
            if np.isnan(imputed_trimmed).any():
                print("Warning: Some NaNs remained after SoftImpute, applying interpolation+filling.")
                imputed_trimmed = pd.Series(imputed_trimmed).interpolate(method='linear', limit_direction='both').fillna(np.nanmean(imputed_trimmed)).to_numpy()

        if n_trimmed < n:
            leftover = series[n_trimmed:]
            imputed_series = pd.Series(np.concatenate([imputed_trimmed, leftover])).interpolate(method='linear', limit_direction='both').fillna(np.nanmean(leftover)).to_numpy()
        else:
            imputed_series = imputed_trimmed

        if np.isnan(imputed_series).any():
            print("Final Warning: NaNs still present after full imputation!")

        return imputed_series

    @staticmethod
    def arima_impute(
        arr: np.ndarray,
    ) -> np.ndarray:
        """
        Impute missing values via ARIMA model forecasting.
        """
        arr = np.asarray(arr)
        original_series = pd.Series(arr)

        temp_series = original_series.interpolate(method='linear', limit_direction='both').fillna(np.nanmean(arr))

        model_auto = auto_arima(temp_series,
                                    stepwise=False,
                                    n_jobs=16,
                                    suppress_warnings=True,
                                    error_action='ignore',
                                    trace=False)

        fitted_values = model_auto.predict_in_sample()

        imputed_arr = np.where(np.isnan(arr), fitted_values, arr)

        return imputed_arr


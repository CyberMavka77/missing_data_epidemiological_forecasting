import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.ticker import FuncFormatter

from simulation_utils import add_noise, mape


def plot_countries_time_series(
    countries_data,
    figsize_per_plot=(12, 3),
    line_color='#004c6d',
    line_width=2):
    """
    Plots each country’s New_cases series in its own subplot (stacked vertically),
    with a rich teal line and quarterly x-axis labels.

    Parameters
    ----------
    countries_data : dict
        Keys are country names, values are DataFrames with 'Date_reported' and 'New_cases'.
    figsize_per_plot : tuple, optional
        (width, height) for each subplot. Total figure height = height * n_countries.
    line_color : str, optional
        Matplotlib color (hex or name) for the line.
    line_width : float, optional
        Thickness of the plot line.
    save_path : str, optional
        If provided, saves the figure to this path (PNG, 300dpi).
    """
    n = len(countries_data)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(figsize_per_plot[0], figsize_per_plot[1] * n),
        sharex=True
    )
    if n == 1:
        axes = [axes]

    def quarter_formatter(x, pos):
        dt = mdates.num2date(x)
        q = (dt.month - 1) // 3 + 1
        return f'Q{q} {dt.year}'

    for ax, (country, df) in zip(axes, countries_data.items()):
        ax.plot(
            df['Date_reported'],
            df['New_cases'],
            color=line_color,
            linewidth=line_width
        )
        ax.set_title(f'New COVID-19 Cases in {country}',
                     fontsize=16, pad=8)
        ax.set_ylabel('New Cases', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.xaxis.set_major_locator(
            mdates.MonthLocator(bymonth=[1,4,7,10])
        )
        ax.xaxis.set_major_formatter(
            FuncFormatter(quarter_formatter)
        )

    bottom_ax = axes[-1]
    bottom_ax.set_xlabel('Date', fontsize=14)
    bottom_ax.tick_params(axis='x', labelrotation=30, labelsize=12)
    bottom_ax.xaxis.set_tick_params(pad=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig, axes


def create_methods_performance_chart(series, country, methods,
                                      noise_perc=0.25, 
                                      nature = "MCAR"):
    """
    Plot imputation results for each method in a single-column stack,
    with bolder lines and high-contrast colors.
    """

    nc = np.array(series['New_cases'].values)
    
    if nature != "MAR":
        curr_series_nc = add_noise(nc, noise_perc, nature=nature)
    else:
        curr_series_nc = add_noise(series, noise_perc, nature=nature)

    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1,
                             figsize=(25, 4 * n_methods),
                             constrained_layout=True)

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, method_func) in zip(axes, methods.items()):
        imputed = method_func(curr_series_nc)
        ax.plot(imputed,
                color='#d62728',
                linewidth=3.0,
                label=f'{method_name} Imputation')
        ax.plot(curr_series_nc,
                color='#1f77b4',
                linewidth=2.5,
                alpha=0.9,
                label='Noisy Data')

        method_mape = mape(nc, imputed)
        ax.set_title(f"{method_name} | MAPE: {100 * method_mape:.2f}%",
                     fontsize=20, pad=10)
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Value', fontsize=18)
        ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
        ax.grid(alpha=0.5, linestyle='--')

    fig.suptitle(f'Imputation Performance by Method — {country}',
                 fontsize=24, y=1.02)


import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


def transfer(data, datetime_col='datetime_beginning_ept', group_col='hour', target_cols=None):
    """
    Generalized data transformation function.
    Converts a datetime column to datetime type, extracts the hour,
    and computes hourly averages for specified target columns.
    The resulting data is rearranged by swapping the first 12 hours with the last 12 hours.
    
    Args:
        data (pd.DataFrame): Input data.
        datetime_col (str): Name of the datetime column to process.
        group_col (str): Name of the column to group by (e.g., 'hour').
        target_cols (list): List of columns to compute averages for.
                            If None, computes averages for all columns.

    Returns:
        np.ndarray: Rearranged array of hourly averaged values.
    """
    # Ensure the datetime column is of datetime type
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Extract hour from the datetime column
    data[group_col] = data[datetime_col].dt.hour
    
    # Compute hourly averages for specified columns (or all columns if not specified)
    if target_cols is not None:
        hourly_avg = data.groupby(group_col)[target_cols].mean()
    else:
        hourly_avg = data.groupby(group_col).mean()
    
    # Convert to array and rearrange (swap first 12 hours with last 12 hours)
    arr = hourly_avg.values
    arr[:12], arr[12:] = arr[12:], arr[:12].copy()

    return arr

def compute_charging_fee():
    """
    Calculate the hourly charging fee rates based on predefined off-peak, super off-peak, and peak prices. 
    The charging fee array is rearranged by swapping the first 12 hours with the last 12 hours.

    Returns:
        list: A list of 24 hourly charging fee rates, rearranged as per the specified swapping logic.
    """
    # Define constants
    aaaaa = 1 / 2
    super_offpeak = 0.16505 * aaaaa
    offpeak = 0.19171 * aaaaa
    peak = 0.38372 * aaaaa

    # Define charging fee schedule (24 hours)
    charging_fee = (
        [offpeak] * 9
        + [super_offpeak] * 5
        + [offpeak] * 2
        + [peak] * 5
        + [offpeak] * 3
    )

    # Rearrange the fee schedule by swapping the first 12 hours with the last 12 hours
    temp = charging_fee[:12].copy()  # Copy the first 12 hours
    charging_fee[:12] = charging_fee[12:]  # Move the last 12 hours to the first half
    charging_fee[12:] = temp  # Place the original first 12 hours in the second half

    return charging_fee

def price_plot(pr_e_rt, pr_fre, charging_fee, mode=0):
    """
    Plots energy price, reserve price, and charging fee with dual y-axes.

    Args:
        pr_e_rt (array-like): Energy price data over time.
        pr_fre (array-like): Reserve price data over time.
        charging_fee (array-like): Charging fee data over time.
        mode (int, optional): If set to 1, saves the plot as a PDF file. Defaults to 0.

    Returns:
        None
    """

    # Initialize the figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 3))
    x = range(len(pr_e_rt))  # Time periods

    # Plot energy and reserve prices on the primary y-axis
    ax1.plot(x, pr_e_rt, alpha=0.6, color='#040676', marker='o', label='Energy Price')
    ax1.plot(x, pr_fre, alpha=0.6, color='#F1B656', marker='v', label='Reserve Price')
    ax1.set_xlim(-0.5, 24.5)
    ax1.set_xlabel('Time Period (1h)')
    ax1.set_ylabel('Market Price ($/kWh)')

    # Adjust x-axis ticks and labels with larger intervals (e.g., every 4 hours)
    major_ticks = range(0, 25, 4)  # Major ticks at every 4 hours
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels([str((i + 12) % 24) for i in major_ticks])  # Circular time labels

    # Create a secondary y-axis for charging fees
    ax2 = ax1.twinx()
    ax2.plot(x, charging_fee, alpha=0.6, color='#A4514F', marker='+', label='Charging Fee')
    ax2.set_ylabel('Charging Fee ($/kWh)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),  # Adjusted position
        ncol=3
    )

    # Save the plot if mode is set to 1
    if mode == 1:
        plt.tight_layout()
        plt.savefig(
            "../output/market_price.pdf",
            bbox_inches='tight',
        )

    # Always show the plot for visualization
    plt.show()

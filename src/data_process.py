import pandas as pd

def transfer(data):
    """
    Process the input data to compute hourly average LMP (Locational Marginal Price) values,
    normalize the values by dividing by 1000, and rearrange the data by swapping the first
    12 hours with the last 12 hours.
    
    Args:
        data (pd.DataFrame): Input dataframe with 'datetime_beginning_ept' (timestamp) and
                             'total_lmp_rt' (real-time LMP) columns.
    
    Returns:
        np.ndarray: A 24-hour array of rearranged hourly average LMP values (in thousands).
    """
    # Convert the timestamp column to datetime format and extract the hour
    data['datetime_beginning_ept'] = pd.to_datetime(data['datetime_beginning_ept']
                                        # format='%Y-%m-%d %H:%M:%S',  # 修改为你的实际日期时间格式
                                        # errors='coerce'  # 如果格式不匹配，将其标记为 NaT
                                        )
    data['hour'] = data['datetime_beginning_ept'].dt.hour

    # Calculate hourly average LMP values and normalize by dividing by 1000
    hourly_avg = data.groupby('hour')[['value']].mean().values / 1000

    # Swap the first 12 hours with the last 12 hours
    hourly_avg[:12], hourly_avg[12:] = hourly_avg[12:], hourly_avg[:12].copy()

    return hourly_avg

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

import time
import numpy as np
import pandas as pd
import os
import random
import torch

def load_climsim_data(root_path = r'H:\My Drive\LEAP_NP_Project\tlnp_supplementary_code\data\climsim_data'):
    # Load the ClimSim Daily Data
    start_time = time.time()

    # Load the long/lat data
    loc_file = os.path.join(root_path, 'location_lat_lon.csv')
    location_dat = pd.read_csv(loc_file)

    climsim_data_path = os.path.join(root_path, 'combined_daily_data_rain.csv')
    climsim_daily_data = pd.read_csv(climsim_data_path)

    # Remove data where rain_rate == 0
    climsim_daily_data = climsim_daily_data[climsim_daily_data['rain_rate'] != 0]

    # Create a new variable called rain_event that is 0 when rain is below the 95th percentile and 1 otherwise
    climsim_daily_data['rain_event'] = np.where(climsim_daily_data['rain_rate'] < np.percentile(climsim_daily_data['rain_rate'], 95), -1, 1)

    # Sort by latitude and then by longitude
    location_dat = location_dat.sort_values(by=['location']).reset_index(drop=True)

    # Divide locations into clusters
    location_dat['cluster_4'] = (location_dat.index // 4)
    location_dat['cluster_8'] = (location_dat.index // 8)
    location_dat['cluster_12'] = (location_dat.index // 12)
    location_dat['cluster_16'] = (location_dat.index // 16)

    # Add location cluster to dataframe
    climsim_daily_data = climsim_daily_data.merge(location_dat[['location', 'cluster_4', 'cluster_8', 'cluster_12', 'cluster_16']], on='location', how='left')

    print(f"Data loaded in {(time.time() - start_time)/60} minutes")

    return climsim_daily_data

def prepare_climsim_data(climsim_data, config, column_name = 'rain_event', seed = None):
    """
    Prepares data for training and testing based on configuration.
    """
    # Set the random seed if provided
    if seed is not None:
        print(f"Setting data random seed to {seed}")
        random.seed(seed)

    # Set target and source data based on config
    data_mode = config["data_mode"]
    target = config["targets"]
    source = config["sources"]

    # Use the specified column based on data_mode to filter target and source data
    target_data = climsim_data[climsim_data[data_mode].isin(target)]
    source_data = climsim_data[climsim_data[data_mode].isin(source)]

    # Shuffle the data
    target_data = target_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    source_data = source_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split into normal and abnormal datasets
    target_normal_data = target_data[target_data[column_name] == -1]
    target_abnormal_data = target_data[target_data[column_name] == 1]
    source_normal_data = source_data[source_data[column_name] == -1]
    source_abnormal_data = source_data[source_data[column_name] == 1]

    # Validate configuration limits with detailed error messages
    if (config["num_target_normal_test"] + config["num_target_normal_training"] > len(target_normal_data)):
        raise ValueError(f"num_target_normal_training ({config['num_target_normal_training']}) + num_target_normal_test ({config['num_target_normal_test']}) cannot be greater than the number of target normal data points ({len(target_normal_data)}).")
    if (config["num_target_abnormal_test"] + config["num_target_abnormal_training"] > len(target_abnormal_data)):
        raise ValueError(f"num_target_abnormal_training ({config['num_target_abnormal_training']}) + num_target_abnormal_test ({config['num_target_abnormal_test']}) cannot be greater than the number of target abnormal data points ({len(target_abnormal_data)}).")
    if config.get("num_source_normal", 0) > len(source_normal_data):
        raise ValueError("Not enough source normal data for the requested split.")
    if config["num_source_abnormal"] > len(source_abnormal_data):
        raise ValueError("Not enough source abnormal data for the requested split.")

    # Train / Test split
    target_normal_train = target_normal_data[:config["num_target_normal_training"]]
    target_normal_test = target_normal_data[-config["num_target_normal_test"]:]
    target_abnormal_train = target_abnormal_data[:config["num_target_abnormal_training"]]
    target_abnormal_test = target_abnormal_data[-config["num_target_abnormal_test"]:]
    source_normal_train   = source_normal_data[:config["num_source_normal"]]
    source_abnormal_train = source_abnormal_data[:config["num_source_abnormal"]]

    # Remove columns and convert to torch tensors
    columns_to_remove = ['date', 'location', 'cluster_4', 'cluster_8', 'cluster_12', 'cluster_16', 'rain_rate', 'snow_rate', 'rain_event', 'snow_event']
    input_dim = config.get('input_dim', 124)
    print(f"Using {input_dim} input features")

    # Drop specified columns and then randomly remove additional columns
    dataframes = [
        target_normal_train, 
        target_abnormal_train, 
        source_normal_train, 
        source_abnormal_train, 
        target_normal_test, 
        target_abnormal_test
    ]
    dropped_dataframes = [df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore') for df in dataframes]

    # Calculate number of random columns to remove and ensure same columns are removed across all
    num_columns_to_remove = dropped_dataframes[0].shape[1] - input_dim
    if num_columns_to_remove > 0:
        columns_to_randomly_remove = random.sample(list(dropped_dataframes[0].columns), num_columns_to_remove)
        dropped_dataframes = [df.drop(columns=columns_to_randomly_remove) for df in dropped_dataframes]

    # Convert to torch tensors
    x0T, x1T, x0S, x1S, x0T_test, x1T_test = [
        torch.tensor(df.values, dtype=torch.float32) for df in dropped_dataframes
    ]

    # Print the number of data points in each dataset
    print(f"Number of data points in target_normal_train: {len(x0T)}")
    print(f"Number of data points in target_abnormal_train: {len(x1T)}")
    print(f"Number of data points in source_normal_train: {len(x0S)}")
    print(f"Number of data points in source_abnormal_train: {len(x1S)}")
    print(f"Number of data points in target_normal_test: {len(x0T_test)}")
    print(f"Number of data points in target_abnormal_test: {len(x1T_test)}")

    return x0T, x1T, x0S, x1S, x0T_test, x1T_test



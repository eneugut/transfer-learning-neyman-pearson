import warnings
import torch
import numpy as np
import random

class TrainingUtils:
    def __init__(self, logger):
        self.logger = logger

    def set_seed(self, seed, all_results):
        if seed:
            self.logger.log(f"Setting seed: {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # Ensure deterministic behavior
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            all_results["seed"] = seed
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def check_error_thresholds(target_normal_dataset, type1_error_lowerbound, type1_error_upperbound):
        n = len(target_normal_dataset)
        min_errors = int(type1_error_lowerbound * n)
        max_errors = int(type1_error_upperbound * n)

        possible_errors = [i for i in range(min_errors, max_errors + 1)
                           if type1_error_lowerbound <= i / n <= type1_error_upperbound]

        if not possible_errors:
            raise ValueError(f"Type 1 error range [{round(type1_error_lowerbound,4)}, {round(type1_error_upperbound,4)}] is not possible with {n} normal samples.")

        elif len(possible_errors) <= 3:
            warnings.warn(f"Type 1 error range [{round(type1_error_lowerbound,4)}, {round(type1_error_upperbound,4)}] has only {len(possible_errors)} possible values: {possible_errors}.", UserWarning)

    @staticmethod
    def standardize_data(data_dict, cols_to_standardize=None):
        # Concatenate all datasets to compute shared mean and std
        combined_data = torch.cat([
            data_dict['target_normal_data'],
            data_dict['target_abnormal_data'],
            data_dict['source_abnormal_data']
        ], dim=0)

        # Compute mean and std from the concatenated dataset
        mean = combined_data.mean(dim=0)
        std = combined_data.std(dim=0)

        # Get the total number of columns
        num_columns = combined_data.size(1)  # This is 0-indexed, so columns go from 0 to num_columns-1

        # If cols_to_standardize is None, standardize all columns
        if cols_to_standardize is None:
            cols_to_standardize = list(range(num_columns))  # Standardize all columns
        else:
            # Check that the specified columns exist (i.e., within the range of available columns)
            for col in cols_to_standardize:
                if col < 0 or col >= num_columns:
                    raise ValueError(f"Column index {col} is out of bounds for {num_columns} columns.")

        # Standardize only the specified columns
        for key, value in data_dict.items():
            if value is not None and value.size(0) > 0:
                # Create a copy to avoid overwriting the original tensor
                value_copy = value.clone()

                # Standardize the specified columns
                for col in cols_to_standardize:
                    value_copy[:, col] = (value[:, col] - mean[col]) / std[col]

                # Update the dictionary with the standardized data
                data_dict[key] = value_copy

    def transfer_to_device(self, device, model, data_dict):
        # Check if the specified device is available, fallback to CPU if necessary
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.log(f"Warning: CUDA is not available. Falling back to CPU.")
            device = 'cpu'
        else:
            self.logger.log(f"Using device: {device}")
        model.to(device)
        for key in data_dict:
            data_dict[key] = data_dict[key].to(device)
        return device

    @staticmethod
    def train_test_split(X, device, validation_split):
        # Shuffle indices
        indices = torch.randperm(X.size(0), device=device)

        # Calculate the test set size
        test_size = int(validation_split * X.size(0))

        # Split the indices
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # Index the data to get training and test sets
        X_train, X_test = X[train_indices], X[test_indices]

        return X_train, X_test

    @staticmethod
    def prepare_data_splits(data_dict, device, validation_split):
        # Split each dataset into training and validation sets
        target_abnormal_train, target_abnormal_val = TrainingUtils.train_test_split(data_dict['target_abnormal_data'], device, validation_split)
        target_normal_train, target_normal_val = TrainingUtils.train_test_split(data_dict['target_normal_data'], device, validation_split)
        source_abnormal_train, source_abnormal_val = TrainingUtils.train_test_split(data_dict['source_abnormal_data'], device, validation_split)

        # Concatenate the training and validation sets
        X_train = torch.cat([target_abnormal_train, target_normal_train, source_abnormal_train], dim=0)
        labels_train = torch.cat([torch.ones(target_abnormal_train.size(0), 1, device=device),
                                    torch.zeros(target_normal_train.size(0), 1, device=device),
                                    2 * torch.ones(source_abnormal_train.size(0), 1, device=device)], dim=0)
        X_val = torch.cat([target_abnormal_val, target_normal_val, source_abnormal_val], dim=0)
        labels_val = torch.cat([torch.ones(target_abnormal_val.size(0), 1, device=device),
                                torch.zeros(target_normal_val.size(0), 1, device=device),
                                2 * torch.ones(source_abnormal_val.size(0), 1, device=device)], dim=0)

        return X_train, labels_train, X_val, labels_val
    
    @staticmethod
    def prepare_evaluation_data(data_dict, device):
        X_evaluation = torch.cat([data_dict['target_abnormal_data'], data_dict['target_normal_data'], data_dict['source_abnormal_data']], dim=0)
        labels_evaluation = torch.cat([
            torch.ones(data_dict['target_abnormal_data'].size(0), device=device),  # Abnormal data gets label 1
            torch.zeros(data_dict['target_normal_data'].size(0), device=device),  # Normal data gets label 0
            2 * torch.ones(data_dict['source_abnormal_data'].size(0), device=device)  # Source data gets label 2
        ], dim=0)
        return X_evaluation, labels_evaluation

    @staticmethod
    def calculate_type1_error_rate(output_normal_class):
        # Calculate the number of false positives
        output_normal_class = torch.sign(output_normal_class)
        false_positives = (output_normal_class == 1).sum().item()
        total_negatives = output_normal_class.numel()

        # Avoid division by zero and calculate Type I error rate
        return false_positives / total_negatives if total_negatives > 0 else 0.0

    @staticmethod
    def calculate_type2_error_rate(output_abnormal_class):
        # Calculate the number of false negatives
        output_abnormal_class = torch.sign(output_abnormal_class)
        false_negatives = (output_abnormal_class == -1).sum().item()
        total_positives = output_abnormal_class.numel()

        # Avoid division by zero and calculate Type II error rate
        return false_negatives / total_positives if total_positives > 0 else 0.0

import torch
import numpy as np
import copy
import math
import random
import time
import traceback
import warnings
import matplotlib.pyplot as plt

class TransferLearningNeymanPearson:
    def __init__(self, model, optimizer, scheduler, loss_function,
                 data_dict,
                 config,
                 seed = None,
                 debug_modes = {
                    'print_training_progress': False,
                    'print_training_epoch_frequency': 100,
                    'print_lr_changes': False,
                    'show_losses_graphs': False,
                    'show_lr_changes': False,
                    'log_scale': True,
                    'print_increment_updates': False,
                    'print_point_selection': False
                 }):
        # Store objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.data_dict = data_dict
        self.seed = seed
        self.debug_modes = debug_modes

        # Clone data to ensure the original data is not modified
        self.data_dict = {key: value.clone() for key, value in data_dict.items()}

        # Unpack config with defaults
        defaults = {
            'num_epochs': 600,
            'batch_size': 512,
            'max_grad_norm': 1.0,
            'early_stopping_patience': 30,
            'early_stopping_min_delta': 0.001,
            'lambda_source_list': [0, 0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 100],
            'method1_constant': 2.5,
            'alpha': 0.2,
            'test_size': 0.2,
            'data_standardization': False,
            'cols_to_standardize': None, # If none, standardizes all columns
            'normalize_losses': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lambda_limit': 1e6,
            'max_tuning_tries': 30,
            'initial_increment_factor': 0.5
        }
        for key, default_value in defaults.items():
            setattr(self, key, config.get(key, default_value))

        # Dict to store all results
        self.all_results = {}
        self.all_model_states = {}

        # Initialization functions
        self._set_seed()
        self._set_lambda_limits()
        self._transfer_to_device()
        self._check_data_dict()

        # Handle optional stage 2 and test data
        self.data_dict.setdefault('target_abnormal_stage2_data', self.data_dict['target_abnormal_data'].clone())
        self.data_dict.setdefault('target_normal_stage2_data', self.data_dict['target_normal_data'].clone())
        self.source_exists = 'source_abnormal_data' in self.data_dict

        # Optional test datasets
        self.has_test_data = any(key in self.data_dict for key in ['target_abnormal_test_data', 'target_normal_test_data', 'secondary_abnormal_test_data'])

        # Set and check error thresholds
        self._set_error_thresholds()
        self._check_error_thresholds()

        # Standardize data
        self._standardize_data()

        # Store initial states
        self._store_initial_states()

        # Ensure lambda_source_list is all floats
        self.lambda_source_list = [float(x) for x in self.lambda_source_list]

    ###########################################################################
    # Setup Functions
    ###########################################################################

    def _set_seed(self):
        if self.seed:
            print(f"Setting seed: {self.seed}")
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            self.all_results["seed"] = self.seed

    def _set_lambda_limits(self):
        if self.lambda_limit > 1:
            self.lambda_max = self.lambda_limit
            self.lambda_min = 1/self.lambda_limit
        else:
            self.lambda_max = 1/self.lambda_limit
            self.lambda_min = self.lambda_limit

    def _transfer_to_device(self):
        # Check if the specified device is available, fallback to CPU if necessary
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'
        else:
            print(f"Using device: {self.device}")
        self.model.to(self.device)
        for key in self.data_dict:
            self.data_dict[key] = self.data_dict[key].to(self.device)

    def _check_data_dict(self):
        # Check if there are missing keys
        if 'target_abnormal_data' not in self.data_dict or 'target_normal_data' not in self.data_dict:
            raise ValueError("data_dict must contain 'target_abnormal_data' and 'target_normal_data'.")

        # Get the number of columns for the first dataset in the dictionary
        first_key = next(iter(self.data_dict))
        num_columns = self.data_dict[first_key].size(1)

        # Iterate over the datasets and check if they have the same number of columns
        for key, tensor in self.data_dict.items():
            if tensor.size(1) != num_columns:
                raise ValueError(f"Dataset '{key}' has {tensor.size(1)} columns, expected {num_columns} columns.")

    def _set_error_thresholds(self):
        epsilon = .5/math.sqrt(len(self.data_dict['target_normal_data']))
        self.type1_error_upperbound = self.alpha + epsilon
        self.type1_error_lowerbound = self.alpha - epsilon

    def _check_error_thresholds(self):
        n = len(self.data_dict['target_normal_stage2_data'])
        min_errors = int(self.type1_error_lowerbound * n)
        max_errors = int(self.type1_error_upperbound * n)

        possible_errors = [i for i in range(min_errors, max_errors + 1)
                           if self.type1_error_lowerbound <= i / n <= self.type1_error_upperbound]

        if not possible_errors:
            raise ValueError(f"Type 1 error range [{round(self.type1_error_lowerbound,4)}, {round(self.type1_error_upperbound,4)}] is not possible with {n} samples.")

        elif len(possible_errors) <= 3:
            warnings.warn(f"Type 1 error range [{round(self.type1_error_lowerbound,4)}, {round(self.type1_error_upperbound,4)}] has only {len(possible_errors)} possible values: {possible_errors}.", UserWarning)

        else:
            print(f"Using Type-I error range: [{round(self.type1_error_lowerbound,4)}, {round(self.type1_error_upperbound,4)}]")

    def _standardize_data(self):
        if self.data_standardization:
            # Concatenate all datasets to compute shared mean and std
            combined_data = torch.cat([
                self.data_dict['target_normal_data'],
                self.data_dict['target_abnormal_data']
            ], dim=0)

            if self.source_exists:
                combined_data = torch.cat([combined_data, self.data_dict['source_abnormal_data']], dim=0)

            # Compute mean and std from the concatenated dataset
            mean = combined_data.mean(dim=0)
            std = combined_data.std(dim=0)

            # Get the total number of columns
            num_columns = combined_data.size(1)  # This is 0-indexed, so columns go from 0 to num_columns-1

            # If cols_to_standardize is None, standardize all columns
            if self.cols_to_standardize is None:
                self.cols_to_standardize = list(range(num_columns))  # Standardize all columns
            else:
                # Check that the specified columns exist (i.e., within the range of available columns)
                for col in self.cols_to_standardize:
                    if col < 0 or col >= num_columns:
                        raise ValueError(f"Column index {col} is out of bounds for {num_columns} columns.")

            # Standardize only the specified columns
            for key, value in self.data_dict.items():
                if value is not None:
                    # Create a copy to avoid overwriting the original tensor
                    value_copy = value.clone()

                    # Standardize the specified columns
                    for col in self.cols_to_standardize:
                        value_copy[:, col] = (value[:, col] - mean[col]) / std[col]

                    # Update the dictionary with the standardized data
                    self.data_dict[key] = value_copy

    def _store_initial_states(self):
        self.initial_model_state = copy.deepcopy(self.model.state_dict())
        self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.initial_scheduler_state = copy.deepcopy(self.scheduler.state_dict()) if self.scheduler else None

    def _restore_initial_states(self):
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        if self.scheduler:
            self.scheduler.load_state_dict(self.initial_scheduler_state)


    ###########################################################################
    # Training Helper Functions
    ###########################################################################

    def _print_epoch_progress(self, epoch, loss, val_loss, total_epoch_time):
        # Calculate the average time per epoch
        average_epoch_time = total_epoch_time / (epoch + 1)  # Add +1 because epoch is zero-indexed

        # Print updates based on your debug settings
        if self.debug_modes.get('print_training_progress', False) and epoch % self.debug_modes["print_training_epoch_frequency"] == 0:
            print(f"Epoch {epoch}/{self.num_epochs}, Training Loss: {round(loss,6)}, Validation Loss: {round(val_loss,6)}, "
                  f"Avg Epoch Time: {round(average_epoch_time, 2)} seconds")

    def _train_test_split(self, X, test_size=0.2):
        # Shuffle indices
        indices = torch.randperm(X.size(0), device=self.device)

        # Calculate the test set size
        test_size = int(test_size * X.size(0))

        # Split the indices
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # Index the data to get training and test sets
        X_train, X_test = X[train_indices], X[test_indices]

        return X_train, X_test

    def _prepare_data_splits(self):
        target_abnormal_train, target_abnormal_val = self._train_test_split(self.data_dict['target_abnormal_data'], test_size=self.test_size)
        target_normal_train, target_normal_val = self._train_test_split(self.data_dict['target_normal_data'], test_size=self.test_size)

        if self.source_exists:
            source_abnormal_train, source_abnormal_val = self._train_test_split(self.data_dict['source_abnormal_data'], test_size=self.test_size)
            X_train = torch.cat([target_abnormal_train, target_normal_train, source_abnormal_train], dim=0)
            labels_train = torch.cat([torch.ones(target_abnormal_train.size(0), 1, device=self.device),
                                      torch.zeros(target_normal_train.size(0), 1, device=self.device),
                                      2 * torch.ones(source_abnormal_train.size(0), 1, device=self.device)], dim=0)
            X_val = torch.cat([target_abnormal_val, target_normal_val, source_abnormal_val], dim=0)
            labels_val = torch.cat([torch.ones(target_abnormal_val.size(0), 1, device=self.device),
                                    torch.zeros(target_normal_val.size(0), 1, device=self.device),
                                    2 * torch.ones(source_abnormal_val.size(0), 1, device=self.device)], dim=0)
        else:
            X_train = torch.cat([target_abnormal_train, target_normal_train], dim=0)
            labels_train = torch.cat([torch.ones(target_abnormal_train.size(0), 1, device=self.device),
                                      torch.zeros(target_normal_train.size(0), 1, device=self.device)], dim=0)
            X_val = torch.cat([target_abnormal_val, target_normal_val], dim=0)
            labels_val = torch.cat([torch.ones(target_abnormal_val.size(0), 1, device=self.device),
                                    torch.zeros(target_normal_val.size(0), 1, device=self.device)], dim=0)
        return X_train, labels_train, X_val, labels_val

    def _update_scheduler_and_learning_rate(self, epoch, val_loss, lr_change_epochs):
        # Update scheduler and learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        # Scheduler step
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

        # Check if the learning rate has changed
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            if self.debug_modes.get('print_lr_changes', False):
                print(f"Epoch {epoch}/{self.num_epochs}: Learning rate changed to {new_lr}")
            lr_change_epochs.append(epoch)

        return lr_change_epochs

    def _check_early_stopping(self, val_loss, best_val_loss, best_model_state, epochs_without_improvement):
        # Early stopping check
        if val_loss < best_val_loss - self.early_stopping_min_delta:
            best_val_loss = val_loss
            best_model_state = self.model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        return best_val_loss, best_model_state, epochs_without_improvement

    def _show_training_loss_graph(self, epoch_training_losses, epoch_validation_losses, lr_change_epochs):
        if self.debug_modes.get('show_losses_graphs', False):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.plot(epoch_training_losses)
            ax1.set_title('Training Loss')
            if self.debug_modes.get('show_lr_changes', False):
                for epoch in lr_change_epochs:
                    ax1.axvline(x=epoch, color='r', linestyle='--', label=f'LR reduced at epoch {epoch}')
            # Apply log scale if specified in debug_modes
            if self.debug_modes.get('log_scale', False):
                ax1.set_yscale('log')
                ax2.set_yscale('log')
            ax2.plot(epoch_validation_losses)
            ax2.set_title('Validation Loss')
            plt.show()

    def _store_lambda_pair_results(self, lambda_source, lambda_normal, epoch_training_losses, epoch_validation_losses, stage2_error_rates, target_abnormal_output_variance):
        # Generate a unique key for this lambda pair
        model_key = f"lambda_source_{lambda_source}_lambda_normal_{lambda_normal}"

        # Store the results for this specific lambda pair
        self.all_results[model_key] = {
            'lambda_source': lambda_source,
            'lambda_normal': lambda_normal,
            'training_losses': epoch_training_losses,
            'validation_losses': epoch_validation_losses,
            'stage2_metrics': {
                'type1_error_rate': stage2_error_rates[0],
                'type2_error_rate_target': stage2_error_rates[1],
                **({'type2_error_rate_source': stage2_error_rates[2]} if len(stage2_error_rates) > 2 else {}),
                'target_abnormal_output_variance': target_abnormal_output_variance
            }
        }

        # Print stage2 results
        output = f"Lambda Source: {lambda_source}, Lambda Normal: {lambda_normal}, Type-I Error Rate: {round(stage2_error_rates[0], 4)}, Type-II Error Rate (Target): {round(stage2_error_rates[1], 4)}"
        if len(stage2_error_rates) > 2:
            output += f", Type-II Error Rate (Source): {round(stage2_error_rates[2], 4)}"
        print(output)

        # Store the model's best state corresponding to this lambda pair for final testing
        if self.has_test_data:
            best_model_state = copy.deepcopy(self.model.state_dict())
            self.all_model_states[model_key] = best_model_state

    def _calculate_type1_error_rate(self, output_normal_class):
        # Calculate the number of false positives
        output_normal_class = torch.sign(output_normal_class)
        false_positives = (output_normal_class == 1).sum().item()
        total_negatives = output_normal_class.numel()

        # Avoid division by zero and calculate Type I error rate
        return false_positives / total_negatives if total_negatives > 0 else 0.0

    def _calculate_type2_error_rate(self, output_abnormal_class):
        # Calculate the number of false negatives
        output_abnormal_class = torch.sign(output_abnormal_class)
        false_negatives = (output_abnormal_class == -1).sum().item()
        total_positives = output_abnormal_class.numel()

        # Avoid division by zero and calculate Type II error rate
        return false_negatives / total_positives if total_positives > 0 else 0.0


    ###########################################################################
    # Core Training Functions
    ###########################################################################

    def train_one_lambda_pair(self, lambda_source, lambda_normal):
        if self.debug_modes.get('print_training_progress', False):
            print(f"Training with lambda_source={lambda_source}, lambda_normal={lambda_normal}")

        # Initialize tracking variables
        epoch_training_losses, epoch_validation_losses, lr_change_epochs = [], [], []
        total_epoch_time = 0
        best_val_loss, best_model_state, epochs_without_improvement = float('inf'), None, 0

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            # Split data into training and validation sets
            X_train, labels_train, X_val, labels_val = self._prepare_data_splits()

            # Train for one epoch
            loss = self._train_one_epoch(X_train, labels_train, lambda_source, lambda_normal)

            # Validate the model
            val_loss = self._validate_model(X_val, labels_val, lambda_source, lambda_normal)

            # Update losses
            epoch_training_losses.append(loss)
            epoch_validation_losses.append(val_loss)

            # End timer for the epoch and calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_epoch_time += epoch_time

            # Print progress if required
            self._print_epoch_progress(epoch, loss, val_loss, total_epoch_time)

            # Learning rate scheduling and early stopping
            lr_change_epochs = self._update_scheduler_and_learning_rate(epoch, val_loss, lr_change_epochs)

            # Check for early stopping
            best_val_loss, best_model_state, epochs_without_improvement = self._check_early_stopping(val_loss, best_val_loss, best_model_state, epochs_without_improvement)
            if epochs_without_improvement >= self.early_stopping_patience:
                if self.debug_modes.get('print_training_progress', False):
                    print(f"Early stopping at epoch {epoch}")
                break

        # Show losses graph if required
        self._show_training_loss_graph(epoch_training_losses, epoch_validation_losses, lr_change_epochs)

        # Restore the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Stage 2 evaluation
        stage2_error_rates, target_abnormal_output_variance = self._evaluate_stage2()

        # Store results
        self._store_lambda_pair_results(lambda_source, lambda_normal, epoch_training_losses, epoch_validation_losses, stage2_error_rates, target_abnormal_output_variance)

        # Restore initial states
        self._restore_initial_states()

        return stage2_error_rates

    def _train_one_epoch(self, X_train, labels_train, lambda_source, lambda_normal):
        self.model.train()
        total_loss = 0

        # Shuffle the dataset
        indices = torch.randperm(X_train.size(0))
        X_train = X_train[indices]
        labels_train = labels_train[indices]
        num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(X_train))
            X_batch = X_train[start_idx:end_idx]
            label_batch = labels_train[start_idx:end_idx]

            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.loss_function(output, label_batch, 1, lambda_source, lambda_normal, self.normalize_losses)
            loss.backward()
            total_loss += loss.item()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

        return total_loss / num_batches

    def _validate_model(self, X_val, labels_val, lambda_source, lambda_normal):
        self.model.eval()
        val_loss = 0

        num_batches_val = (len(X_val) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for i in range(num_batches_val):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X_val))
                X_batch_val = X_val[start_idx:end_idx]
                label_batch_val = labels_val[start_idx:end_idx]

                output_val = self.model(X_batch_val)
                val_loss += self.loss_function(output_val, label_batch_val, 1, lambda_source, lambda_normal, self.normalize_losses).item()

        self.model.train()
        return val_loss / num_batches_val

    def _evaluate_stage2(self):
        # Stage 2 evaluation with batching
        output_stage2 = []

        self.model.eval()
        with torch.no_grad():
            # Concatenate test data
            if self.source_exists:
                X_stage2 = torch.cat([self.data_dict['target_abnormal_stage2_data'], self.data_dict['target_normal_stage2_data'], self.data_dict['source_abnormal_data']], dim=0)
                labels_stage2 = torch.cat([
                    torch.ones(self.data_dict['target_abnormal_stage2_data'].size(0), device=self.device),  # Abnormal data gets label 1
                    torch.zeros(self.data_dict['target_normal_stage2_data'].size(0), device=self.device),  # Normal data gets label 0
                    2 * torch.ones(self.data_dict['source_abnormal_data'].size(0), device=self.device)  # Source data gets label 2
                ], dim=0)
            else:
                X_stage2 = torch.cat([self.data_dict['target_abnormal_stage2_data'], self.data_dict['target_normal_stage2_data']], dim=0)
                labels_stage2 = torch.cat([
                    torch.ones(self.data_dict['target_abnormal_stage2_data'].size(0), device=self.device),  # Abnormal data gets label 1
                    torch.zeros(self.data_dict['target_normal_stage2_data'].size(0), device=self.device)    # Normal data gets label 0
                ], dim=0)

            num_batches = (len(X_stage2) + self.batch_size - 1) // self.batch_size
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X_stage2))
                X_batch_stage2 = X_stage2[start_idx:end_idx]
                label_batch_stage2 = labels_stage2[start_idx:end_idx]

                output_stage2.append(self.model(X_batch_stage2))

        output_stage2 = torch.cat(output_stage2, dim=0)
        self.model.train()

        # Calculate error rates for Stage 2
        type1_error_rate = self._calculate_type1_error_rate(output_stage2[labels_stage2 == 0])
        type2_error_rate_target = self._calculate_type2_error_rate(output_stage2[labels_stage2 == 1])

        # Calculate variance in target abnormal outputs
        target_abnormal_output_variance = torch.var(torch.sign(output_stage2[labels_stage2 == 1]).float()).item()

        if self.source_exists:
            type2_error_rate_source = self._calculate_type2_error_rate(output_stage2[labels_stage2 == 2])

        if self.source_exists:
            return (type1_error_rate, type2_error_rate_target, type2_error_rate_source), target_abnormal_output_variance
        return (type1_error_rate, type2_error_rate_target), target_abnormal_output_variance

    def _evaluate_on_test_data(self, best_point, print_output_string = True):
        output_string = f"\nEvaluation on test data:\n"
        test_error_dict = {}
        # Load the best model state for the best point
        best_model_state = self.all_model_states[f"lambda_source_{best_point[0]}_lambda_normal_{best_point[1]}"]
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        with torch.no_grad():
            type1_error_test, type2_error_test, secondary_type2_error_test = None, None, None
            test_datasets = ['target_normal_test_data', 'target_abnormal_test_data', 'secondary_abnormal_test_data']
            for test_dataset in test_datasets:
                if test_dataset in self.data_dict:
                    # Evaluate on each of the test sets. Add to the dict and the output_string
                    output_test = self.model(self.data_dict[test_dataset])
                    if test_dataset == 'target_normal_test_data':
                        type1_error_test = self._calculate_type1_error_rate(output_test)
                        test_error_dict["type1_error_test"] = type1_error_test
                        output_string += f"     Type-I Error Rate: {type1_error_test}\n"
                    elif test_dataset == 'target_abnormal_test_data':
                        type2_error_test = self._calculate_type2_error_rate(output_test)
                        test_error_dict["type2_error_test"] = type2_error_test
                        output_string += f"     Type-II Error Rate: {type2_error_test}\n"
                    else:
                        secondary_type2_error_test = self._calculate_type2_error_rate(output_test)
                        test_error_dict["secondary_type2_error_test"] = secondary_type2_error_test
                        output_string += f"     Secondary Type-II Error Rate: {secondary_type2_error_test}\n"
            print(output_string)

        # Reset initial states
        self._restore_initial_states()

        return test_error_dict

    ###########################################################################
    # Core Process Functions
    ###########################################################################

    def run_training_process(self):
        try:
            if self.source_exists:
                self.run_training_with_source()
            else:
                self.run_training_without_source()
            return self.all_results

        except Exception as e:
            print(f"Error occurred during training process: {e}")
            print(traceback.format_exc())

    def run_training_without_source(self):
        print(f"No source data detected. Training without source.")
        best_point = self.fine_tune_lambda(lambda_source=0, lambda_normal=1, tune_lambda="normal")
        if not best_point:
            best_point = self._find_best_point_after_failures()
        if not best_point:
            raise ValueError(f"No suitable point could be found.")
        test_error_dict = self._evaluate_on_test_data(best_point)
        self.all_results["test_metrics"] = {
            'best_lambda_source': best_point[0],
            'best_lambda_normal': best_point[1],
            **test_error_dict,
            'num_trainings': len([key for key in self.all_results if key.startswith("lambda")])
        }

    def run_training_with_source(self):
        low_magnitude = min([x for x in self.lambda_source_list if x != 0])
        high_magnitude = max(self.lambda_source_list)
        lambda_normal = 1.0
        converged_lambda_source_list = set()
        num_tuned_points = 0

        while self.lambda_source_list:
            lambda_source = self.lambda_source_list.pop(0)
            tuned_point = self.fine_tune_lambda(lambda_source, lambda_normal, tune_lambda='normal')
            if tuned_point:
                lambda_normal = tuned_point[1]
                num_tuned_points += 1
                converged_lambda_source_list.add(lambda_source)
            self._update_lambda_list(low_magnitude, high_magnitude, num_tuned_points)

        num_trainings = len([key for key in self.all_results if key.startswith("lambda")])
        print(f"Number of trainings in process: {num_trainings}")

        # Select the final lambda pair classifier
        print(f"\nSelecting final point...")

        # Filter points by Type-I Error threshold
        filtered_points = self._filter_points_by_type1_error(self.type1_error_upperbound)
        num_points_threshold1 = len(filtered_points)
        if self.debug_modes.get('print_point_selection', False):
            print(f"Points below Type-I threshold of {round(self.type1_error_upperbound,4)}: {num_points_threshold1}\n")

        # Evaluate and store results
        final_point_method1, num_points_threshold2_method1, target_type2_upperbound_method1 = self._choose_final_point_method1(filtered_points)
        self._evaluate_and_store_results(
            method_name="method1",
            final_point=final_point_method1,
            num_points_threshold1=num_points_threshold1,
            num_points_threshold2=num_points_threshold2_method1,
            type1_upperbound=self.type1_error_upperbound,
            target_type2_upperbound=target_type2_upperbound_method1
        )

        self.all_results['num_trainings'] = num_trainings
        if self.seed is not None:
            self.all_results["seed"] = self.seed

    ###########################################################################
    # Helper Functions for Core Process
    ###########################################################################

    def _update_lambda_list(self, low_magnitude, high_magnitude, num_tuned_points):
        if len(self.lambda_source_list) == 0 and num_tuned_points < 5:
            low_magnitude -= 1
            high_magnitude += 1
            if 10**low_magnitude > self.lambda_min:
                self.lambda_source_list.append(10**low_magnitude)
            if 10**high_magnitude < self.lambda_max:
                self.lambda_source_list.append(10**high_magnitude)

    def _choose_final_point_method1(self, filtered_points):
        # Sort and choose best point based on type2_error_rate_target
        best_point = min(filtered_points, key=lambda p: p['stage2_metrics']['type2_error_rate_target'])
        if self.debug_modes.get('print_point_selection', False):
            print(f"Best point by Type-II Error (Target):")
            self._print_point_and_stage2_metrics(best_point)

        # Calculate Type II error upperbound for the target
        target_type2_upperbound = best_point['stage2_metrics']['type2_error_rate_target'] + (self.method1_constant * (1 / math.sqrt(self.data_dict["target_abnormal_stage2_data"].size(0))))

        # Further filter points based on Type-II error rate (target)
        final_filtered_points = [p for p in filtered_points if p['stage2_metrics']['type2_error_rate_target'] <= target_type2_upperbound]
        num_points_threshold2 = len(final_filtered_points)
        if self.debug_modes.get('print_point_selection', False):
            print(f"Points below Type-II (Target) threshold of {round(target_type2_upperbound,4)}: {num_points_threshold2}")

        # Choose the point with the lowest type2_error_rate_source
        final_point = self._find_point_with_lowest_type2_error_source(final_filtered_points)
        if self.debug_modes.get('print_point_selection', False):
            print(f"Final point:")
            self._print_point_and_stage2_metrics(final_point)

        return final_point, num_points_threshold2, target_type2_upperbound

    def _filter_points_by_type1_error(self, error_threshold):
        filtered_points = []
        for key, result in self.all_results.items():
            try:
                if result['stage2_metrics']['type1_error_rate'] <= error_threshold:
                    filtered_points.append(result)
            except:
                pass
        return filtered_points

    def _find_point_with_lowest_type2_error_source(self, filtered_points):
        return min(
            filtered_points,
            key=lambda p: (
                p['stage2_metrics']['type2_error_rate_source'],
                p['stage2_metrics']['type2_error_rate_target'],
                p['stage2_metrics']['type1_error_rate']
            )
        )

    def _print_point_and_stage2_metrics(self, point):
        output_string = f"      Lambda Source: {point['lambda_source']}\n      Lambda Normal: {point['lambda_normal']}\n"
        output_string += f"      Stage 2 Type-I Error: {point['stage2_metrics']['type1_error_rate']}\n"
        output_string += f"      Stage 2 Type-II Error (Target): {point['stage2_metrics']['type2_error_rate_target']}\n"
        output_string += f"      Stage 2 Type-II Error (Source): {point['stage2_metrics']['type2_error_rate_source']}"
        print(output_string)

    def _evaluate_and_store_results(self, method_name, final_point, num_points_threshold1, num_points_threshold2, type1_upperbound, target_type2_upperbound):
        # Evaluate the final model on the test dataset
        final_point = (final_point["lambda_source"], final_point["lambda_normal"])
        test_error_dict = self._evaluate_on_test_data(final_point)

        # Store results in self.all_results
        result_entry = {
            'best_lambda_source': final_point[0],
            'best_lambda_normal': final_point[1],
            **test_error_dict,
            'type1_upperbound': type1_upperbound,
            'num_points_threshold1': num_points_threshold1,
            'target_type2_upperbound': target_type2_upperbound,
            'num_points_threshold2': num_points_threshold2
        }

        self.all_results[f"test_metrics_{method_name}"] = result_entry

    def _find_best_point_after_failures(self):
        # Filter all results to include only points with Type I error below the upper bound and that have 'stage2_metrics'
        eligible_points = [(key, value) for key, value in self.all_results.items()
                          if isinstance(value, dict) and 'stage2_metrics' in value and value['stage2_metrics']['type1_error_rate'] < self.type1_error_upperbound]

        if not eligible_points:
            # If no points meet the Type I error threshold, consider all points that have 'stage2_metrics'
            eligible_points = [(key, value) for key, value in self.all_results.items()
                              if isinstance(value, dict) and 'stage2_metrics' in value]

        # If still no eligible points found, raise an error
        if not eligible_points:
            return None

        # Sort points by Type II error rate, then Type I error rate, then order of training
        eligible_points.sort(key=lambda x: (
            x[1]['stage2_metrics']['type2_error_rate_target'],
            x[1]['stage2_metrics']['type1_error_rate']
        ))

        # Return the best point based on sorting criteria
        best_point_key, best_point_value = eligible_points[0]
        return (best_point_value['lambda_source'], best_point_value['lambda_normal'],
                best_point_value['stage2_metrics']['type1_error_rate'],
                best_point_value['stage2_metrics']['type2_error_rate_target'])

    ###########################################################################
    # Fine-Tuning Lambda Functions
    ###########################################################################

    def fine_tune_lambda(self, lambda_source, lambda_normal, prior_type1_error=None, tune_lambda='normal'):
        if not prior_type1_error:
            prior_type1_error = (self.type1_error_lowerbound + self.type1_error_upperbound) / 2

        if tune_lambda not in ['normal', 'source']:
            raise ValueError("tune_lambda must be either 'normal' or 'source'")

        print(f"Fixing lambda_source at {lambda_source} and fine tuning lambda_normal (starting with {lambda_normal}).") if tune_lambda == 'normal' else print(f"Fixing lambda_normal at {lambda_normal} and Fine tuning lambda_source (starting with {lambda_source}).")

        increment_factor = self.initial_increment_factor
        tries, tries_no_overshooting = 0, 0

        while tries < self.max_tuning_tries:
            stage2_error_rates = self.train_one_lambda_pair(lambda_source, lambda_normal)
            type1_error, type2_error = stage2_error_rates[0], stage2_error_rates[1]

            if self._is_within_error_bounds(type1_error):
                return lambda_source, lambda_normal, type1_error, type2_error

            increment_factor, tries_no_overshooting = self._adjust_increment_factor(
                prior_type1_error, type1_error, increment_factor, tries_no_overshooting
            )

            lambda_source, lambda_normal = self._adjust_lambdas(
                lambda_source, lambda_normal, increment_factor, type1_error, tune_lambda
            )

            if self._is_lambda_out_of_bounds(lambda_source, lambda_normal, tune_lambda):
                print(f"Lambda_{tune_lambda} of {lambda_normal if tune_lambda == 'normal' else lambda_source} too small or too large. Lambda did not converge.")
                break

            prior_type1_error = type1_error
            tries += 1

        if tries == self.max_tuning_tries:
            print(f"Max tries of {self.max_tuning_tries} reached. Lambda did not converge.")
        return None

    def _adjust_lambdas(self, lambda_source, lambda_normal, increment_factor, type1_error, tune_lambda):
        if tune_lambda == 'normal':
            lambda_normal = lambda_normal * (1 + increment_factor) if type1_error > self.type1_error_upperbound else lambda_normal * (1 - increment_factor)
        else:
            lambda_source = lambda_source * (1 + increment_factor) if type1_error <= self.type1_error_upperbound else lambda_source * (1 - increment_factor)

        if self.debug_modes.get('print_increment_updates', False):
            print(f"Applying increment_factor of {1 + increment_factor if type1_error > self.type1_error_upperbound else 1 - increment_factor} to lambda_{tune_lambda}.")

        return lambda_source, lambda_normal

    def _adjust_increment_factor(self, prior_type1_error, type1_error, increment_factor, tries_no_overshooting):
        if (prior_type1_error <= self.type1_error_lowerbound and type1_error >= self.type1_error_upperbound) or \
          (prior_type1_error >= self.type1_error_upperbound and type1_error <= self.type1_error_lowerbound):
            increment_factor /= 2
            tries_no_overshooting = 0
            if self.debug_modes.get('print_increment_updates', False):
                print(f"Over/undershooting. Halving increment_factor to {increment_factor}.")
        else:
            tries_no_overshooting += 1
            if tries_no_overshooting >= 5:
                increment_factor = (1 - increment_factor) / 2 + increment_factor
                if self.debug_modes.get('print_increment_updates', False):
                    print(f"5+ attempts without over/undershooting. Increasing increment_factor to {increment_factor}.")

        return increment_factor, tries_no_overshooting

    def _is_within_error_bounds(self, type1_error):
        return self.type1_error_lowerbound <= type1_error <= self.type1_error_upperbound

    def _is_lambda_out_of_bounds(self, lambda_source, lambda_normal, tune_lambda):
        return (tune_lambda == 'normal' and (lambda_normal < self.lambda_min or lambda_normal > self.lambda_max)) or \
              (tune_lambda == 'source' and (lambda_source < self.lambda_min or lambda_source > self.lambda_max))

import torch
import copy
import math
import time
import traceback
import json

from tlnp_lib.lambda_tuner import LambdaTuner
from tlnp_lib.training_logger import TrainingLogger
from tlnp_lib.training_utils import TrainingUtils
from tlnp_lib.point_selection_utils import PointSelectionUtils
from tlnp_lib.max_risk_objective import MaxRiskObjective
from tlnp_lib.transfer_learning_neyman_pearson import TransferLearningNeymanPearson

class NewTransferLearningNeymanPearson(TransferLearningNeymanPearson):
    def __init__(self, config, data_dict, model, loss_function, optimizer, scheduler):
        # Unpack config with defaults
        defaults = {
            'num_epochs': 100,
            'batch_size': 16,
            'max_grad_norm': None,
            'early_stopping_patience': 30,
            'early_stopping_min_delta': 0.001,
            'lambda_source_list': [0, 0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 100],
            'selection_constant': 0.5,
            'type1_error_upperbound': 0.2,
            'type1_error_lowerbound': None,
            'constant_target_normal': 1,
            'constant_target_abnormal': 1,
            'constant_source_normal': 1,
            'constant_source_abnormal': 1,
            'eta_alpha': 1,
            'eta_theta': 1,
            'eta_lambda': 1,
            'gamma': 1,
            'main_training_loss_function_type': "ExponentialLoss",
            'validation_split': 0.2,
            'data_standardization': False,
            'cols_to_standardize': None,  # If none, standardizes all columns
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lambda_limit': 1e6,
            'max_tuning_tries': 30,
            'initial_increment_factor': 0.5,
            'seed': None,
            'model_save_path': None,
            'results_save_path': None,
            'restore_model_after_completion': True,
        }

        super().__init__(config, data_dict, model, loss_function, optimizer, scheduler, defaults)

    ###########################################################################
    # Setup Functions
    ###########################################################################

    def _set_epsilons(self):
        self.epsilon_0_T = self.constant_target_normal / math.sqrt(len(self.data_dict['target_normal_data']))
        self.epsilon_1_T = self.constant_target_abnormal / math.sqrt(len(self.data_dict['target_abnormal_data']))
        if self.source_exists:
            self.epsilon_0_S = self.constant_source_normal / math.sqrt(len(self.data_dict['source_normal_data']))
            self.epsilon_1_S = self.constant_source_abnormal / math.sqrt(len(self.data_dict['source_abnormal_data']))
        else:
            self.epsilon_0_S = 0
            self.epsilon_1_S = 0

    # Override the base class method to set Type-I error bounds with epsilon
    def _set_type1_lowerbound(self):
        self._set_epsilons()
        
        if self.type1_error_upperbound <= 0 or self.type1_error_upperbound >= 1:
            raise ValueError(
                f"Type-I error upperbound must be between 0 and 1.")
        if not self.type1_error_lowerbound:
            self.type1_error_lowerbound = self.type1_error_upperbound - self.epsilon_0_T * 2
        else:
            if self.type1_error_lowerbound < 0 or self.type1_error_lowerbound >= self.type1_error_upperbound:
                raise ValueError(
                    f"Type-I error lowerbound must be between 0 and {self.type1_error_upperbound}.")
        self.logger.log(
            f"Using Type-I error range: [{round(self.type1_error_lowerbound,4)}, {round(self.type1_error_upperbound,4)}]")
        self.alpha = (self.type1_error_lowerbound + self.type1_error_upperbound)/2

    ###########################################################################
    # Training Helper Functions
    ###########################################################################

    def _safe_num_batches(self, labels: torch.Tensor):
        # counts for codes 0..3
        counts = [int((labels.view(-1).long() == k).sum()) for k in (0,1,2,3)]
        min_per_group = max(1, min(counts))  # at least 1 if groups exist
        target_batches = max(1, int(round(labels.size(0) / float(self.batch_size))))
        return min(target_batches, min_per_group)

    def _batch_iterator(
        self, X: torch.Tensor, labels: torch.Tensor, num_batches: int,
        enforce_presence: bool = True, shuffle_within_batch: bool = True
    ):
        y = labels.view(-1).long()
        groups = [torch.where(y == k)[0] for k in (0, 1, 2, 3)]

        # shuffle within each group (once per epoch)
        for k in range(4):
            if groups[k].numel() > 0:
                perm = torch.randperm(groups[k].numel(), device=groups[k].device)
                groups[k] = groups[k][perm]

        if enforce_presence:
            for k in range(4):
                n = groups[k].numel()
                if n == 0:
                    raise ValueError(f"Group {k} has 0 samples; cannot include it in every batch.")
                if n < num_batches:
                    raise ValueError(
                        f"Group {k} has only {n} samples but num_batches={num_batches}; "
                        "reduce num_batches or set enforce_presence=False."
                    )

        # build per-group slice plan
        plans = []
        for k in range(4):
            idx = groups[k]
            n = idx.numel()
            base, rem = (n // num_batches), (n % num_batches)
            sizes = [(base + 1 if b < rem else base) for b in range(num_batches)]
            s, slices = 0, []
            for sz in sizes:
                slices.append((s, s + sz))
                s += sz
            assert s == n, "Slice planning error—group not fully covered."
            plans.append(slices)

        # emit batches
        for b in range(num_batches):
            parts = []
            for k in range(4):
                s, t = plans[k][b]
                if s < t:
                    parts.append(groups[k][s:t])
            if not parts:
                continue
            batch_idx = torch.cat(parts)
            if shuffle_within_batch:
                perm = torch.randperm(batch_idx.numel(), device=batch_idx.device)
                batch_idx = batch_idx[perm]
            yield X[batch_idx], y[batch_idx]
            
    ###########################################################################
    # Core Training Functions
    ###########################################################################

    def run_main_model_training(self):
        self.logger.log_training_progress(f"Running main model training")
        self.alpha_prime_list = []

        # Initialize tracking variables
        epoch_training_losses, epoch_validation_losses, lr_change_epochs = [], [], []
        total_epoch_time = 0
        best_val_loss, best_model_state, epochs_without_improvement = float(
            'inf'), None, 0

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            # Split data into training and validation sets
            X_train, labels_train, X_val, labels_val = self.utils.prepare_data_splits(
                self.data_dict, self.device, self.validation_split)

            # Train for one epoch
            loss = self._train_one_epoch_main(X_train, labels_train)

            # Validate the model
            val_loss = self._validate_model_main(X_val, labels_val)

            # Update losses
            epoch_training_losses.append(loss)
            epoch_validation_losses.append(val_loss)

            # End timer for the epoch and calculate epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_epoch_time += epoch_time

            # Print progress
            self.logger.log_training_progress_losses(
                epoch, loss, val_loss, total_epoch_time, self.num_epochs)

            # Learning rate scheduling and early stopping
            lr_change_epochs = self._update_scheduler_and_learning_rate(
                epoch, val_loss, lr_change_epochs)

            # Check for early stopping
            best_val_loss, best_model_state, epochs_without_improvement = self._check_early_stopping(
                val_loss, best_val_loss, best_model_state, epochs_without_improvement)
            if epochs_without_improvement >= self.early_stopping_patience:
                self.logger.log_training_progress(
                    f"Early stopping at epoch {epoch}")
                break

        # Show losses graph if required
        self.logger.show_training_loss_plot(
            epoch_training_losses, epoch_validation_losses, lr_change_epochs)

        # Restore the best model state for evaluation
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Lambda pair evaluation
        evaluation_error_rates = self._evaluate_lambda_pair()

        # Evaluate alpha_prime statistics
        avg_alpha_prime = (
            sum(self.alpha_prime_list) / len(self.alpha_prime_list)
            if self.alpha_prime_list else float("nan")
        )
        self.logger.log_training_progress(
            f"Average alpha' across training steps: {avg_alpha_prime:.6f}"
        )
        print(self.alpha_prime_list)

        # Store results
        self._store_main_training_results(epoch_training_losses,
            epoch_validation_losses, evaluation_error_rates, avg_alpha_prime, self.alpha_prime_list)

        return evaluation_error_rates

    def _train_one_epoch_main(self, X_train, labels_train):
        self.model.train()
        total_g, steps = 0.0, 0

        num_batches = self._safe_num_batches(labels_train)
        for Xb, Yb in self._batch_iterator(X_train, labels_train, num_batches, enforce_presence=True):
            # Zero grads
            self.optimizer.zero_grad(set_to_none=True)
            if self.obj.alpha_prime.grad is not None:
                self.obj.alpha_prime.grad.zero_()

            # Forward -> g(θ_t, α'_t)
            try:
                g, _, (f1, f2, f3), (R0T, R1T, R0S) = self.obj(Xb, Yb)
            except ValueError:
                continue

            # Backprop
            g.backward()

            # θ step scaled by λ_t
            with torch.no_grad():
                lam = self.obj.lmbda
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(lam)

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # α′ step + projection
            with torch.no_grad():
                grad_alpha = self.obj.alpha_prime.grad
                if grad_alpha is None:
                    grad_alpha = torch.tensor(0.0, device=self.obj.alpha_prime.device)
                self.obj.alpha_prime -= self.eta_alpha * (1.0 + self.obj.lmbda * grad_alpha)
                self.obj.alpha_prime.data.clamp_(min=self.obj.alpha)
                self.obj.alpha_prime.grad = None
                
                # record α′ after the update
                self.alpha_prime_list.append(float(self.obj.alpha_prime.detach().item()))

            # λ step
            with torch.no_grad():
                self.obj.lmbda = torch.relu(
                    (1 - self.gamma * self.eta_lambda) * self.obj.lmbda
                    + self.eta_lambda * g.detach()
                )

            total_g += g.detach().item()
            steps += 1

        return total_g / max(steps, 1)

    def _validate_model_main(self, X_val, labels_val):
        self.model.eval()
        total_g, steps = 0.0, 0
        with torch.no_grad():
            num_batches = self._safe_num_batches(labels_val)
            for Xb, Yb in self._batch_iterator(X_val, labels_val, num_batches, enforce_presence=True):
                g, _, _, _ = self.obj(Xb, Yb)
                total_g += g.item()
                steps += 1
        self.model.train()
        return total_g / max(steps, 1)

    def _store_main_training_results(self, epoch_training_losses, epoch_validation_losses, evaluation_error_rates, alpha_prime_avg, alpha_prime_list):
        # Generate a unique key for this lambda pair
        # Store the results for this specific lambda pair
        self.all_results["main_training_results"] = {
            'training_losses': epoch_training_losses,
            'validation_losses': epoch_validation_losses,
            'evaluation_metrics': {
                'type1_error_rate': evaluation_error_rates[0],
                'type2_error_rate_target': evaluation_error_rates[1],
                **({'type2_error_rate_source': evaluation_error_rates[2]} if len(evaluation_error_rates) > 2 else {}),
            },
            'alpha_prime_avg': alpha_prime_avg,
            'alpha_prime_list': alpha_prime_list,
        }

        # Print evaluation results
        output = f"""
        Main Training Results: 
        Type-I Error Rate: {evaluation_error_rates[0]:.4f}
        Type-II Error Rate (Target): {evaluation_error_rates[1]:.4f}
        Type-II Error Rate (Source): {round(evaluation_error_rates[2], 4)}
        """
        self.logger.log_training_progress(output)

        # Store the model's best state corresponding to this lambda pair for final testing
        if self.has_test_data:
            self.main_model_state = copy.deepcopy(self.model.state_dict())

    def _evaluate_on_test_data_main(self):
        # Evaluate test data after all trainings have completed and the final point is chosen        
        test_error_dict = {}
        if not self.has_test_data:
            return test_error_dict
        output_string = f"\nEvaluation on test data:\n"

        # Evaluate on test data
        self.model.eval()
        with torch.no_grad():
            type1_error_test, type2_error_test = None, None
            test_datasets = ['target_normal_test_data',
                             'target_abnormal_test_data']

            # Evaluate on each of the test sets. Add results to the dict and log results
            for test_dataset in test_datasets:
                if test_dataset in self.data_dict:
                    output_test = self.model(self.data_dict[test_dataset])

                    # Subtract the threshold if it exists (for Naive NP)
                    if hasattr(self, "optimal_threshold"):
                        output_test = output_test - self.optimal_threshold
                        output_string += f"Threshold: {self.optimal_threshold}\n"

                    # Calculate Type-I Error from the target normal test data
                    if test_dataset == 'target_normal_test_data':
                        type1_error_test = self.utils.calculate_type1_error_rate(
                            output_test)
                        test_error_dict["type1_error_test"] = type1_error_test
                        output_string += f"     Type-I Error Rate: {type1_error_test}\n"

                    # Calculate Type-II Error from the target abnormal test data
                    elif test_dataset == 'target_abnormal_test_data':
                        type2_error_test = self.utils.calculate_type2_error_rate(
                            output_test)
                        test_error_dict["type2_error_test"] = type2_error_test
                        output_string += f"     Type-II Error Rate: {type2_error_test}\n"
            self.logger.log(output_string)

        # Save the model
        if self.model_save_path:
            model_save_path = self.model_save_path + self.model_suffix + ".pth"
            self.logger.log(f"Saving model to {model_save_path}")
            torch.save(self.model.state_dict(), model_save_path)

        # Reset initial states in case of further training
        if self.restore_model_after_completion:
            self._restore_initial_states()

        return test_error_dict
        
    ###########################################################################
    # Core Process Functions
    ###########################################################################

    def run_training_process(self):
        try:
            start_time = time.time()
            
            self.run_training_with_source()
                
            process_time = time.time() - start_time
            self.all_results['process_time'] = process_time

            if self.results_save_path:
                with open(self.results_save_path + '_tlnp.json', 'w') as f:
                    json.dump(self.all_results, f, indent=4)
            return self.all_results

        except Exception as e:
            print(f"Error occurred during training process: {e}")
            print(traceback.format_exc())

    def compute_R1T_hat_const(self):
        # Load best model state
        best_lambda_normal = self.all_results['test_metrics']['best_lambda_normal']
        best_model = self.all_model_states[f'lambda_source_0.0_lambda_normal_{best_lambda_normal}']
        self.model.load_state_dict(best_model)

        # Get all training data (no val split)
        X_all, labels_all, _, _ = self.utils.prepare_data_splits(self.data_dict, self.device, validation_split=0.0)
        # Compute R1T_hat_const
        R1T_hat_const = MaxRiskObjective.compute_R1T_hat_const_from_data(
            self.model,
            X_all,
            labels_all,
            code_T1=1,                                          # your label code for T1
            loss_function_type=self.main_training_loss_function_type,    # "LogisticLoss" / "ExponentialLoss" / "HingeLoss"
            device=self.device,
        )
        
        # Restore model state
        self._restore_initial_states()

        return R1T_hat_const

    def run_training_with_source(self):
        # First, run training without source
        self.data_dict_copy = copy.deepcopy(self.data_dict)
        feature_dim = self.data_dict['target_abnormal_data'].shape[1]
        self.data_dict = {
            'target_normal_data': self.data_dict_copy['target_normal_data'],
            'target_abnormal_data': self.data_dict_copy['target_abnormal_data'],
            'source_normal_data': torch.empty((0, feature_dim)),
            'source_abnormal_data': torch.empty((0, feature_dim)),
        }
        self.run_training_without_source()

        # compute R1T_hat_const once (with theta_hat model) and pass it in
        self.obj = MaxRiskObjective(
            model=self.model,
            alpha=self.alpha,
            epsilon_0_T=self.epsilon_0_T, epsilon_1_T=self.epsilon_1_T,
            epsilon_0_S=self.epsilon_0_S, epsilon_1_S=self.epsilon_1_S,
            R1T_hat_const=self.compute_R1T_hat_const(),
            loss_function_type=self.main_training_loss_function_type,
        ).to(self.device)

        # Restore data dict
        self.data_dict = self.data_dict_copy
        # Dict to store all results and model states
        self.all_results = {
            'approach_name': 'new_tlnp', 
            'config': copy.deepcopy(self.all_results["config"]), 
            'first_stage_results': copy.deepcopy(self.all_results)
        }
        
        # Run main stage training process
        self.run_main_model_training()
        
        # Compute and store results on test data
        self.all_results[f"test_metrics"] = self._evaluate_on_test_data_main()

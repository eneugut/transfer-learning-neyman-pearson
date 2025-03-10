import torch

from tlnp.transfer_learning_neyman_pearson import TransferLearningNeymanPearson

class NaiveNeymanPearson(TransferLearningNeymanPearson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_training_process(self):
        lambda_source = 1
        lambda_normal = 1
        self.train_one_lambda_pair(lambda_source, lambda_normal)

        test_error_dict = self._evaluate_on_test_data((lambda_source, lambda_normal))
        self.all_results["test_metrics"] = {
            **test_error_dict,
            'num_trainings': len([key for key in self.all_results if key.startswith("lambda")])
        }

        return self.all_results

    ###########################################################################

    def _evaluate_stage2(self):
        # Stage 2 evaluation with batching
        self.model.eval()
        output_stage2 = []

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

        # Find threshold to meet Type I Error constraints
        self.optimal_threshold = self._calculate_optimal_threshold(output_stage2, labels_stage2)
        self.all_results["optimal_threshold"] = self.optimal_threshold

        # Adjust outputs based on threshold
        output_stage2 = output_stage2 - self.optimal_threshold

        # Calculate error rates for Stage 2
        type1_error_rate = self._calculate_type1_error_rate(output_stage2[labels_stage2 == 0])
        type2_error_rate_target = self._calculate_type2_error_rate(output_stage2[labels_stage2 == 1])

        # Calculate variance in target abnormal outputs
        target_abnormal_output_variance = torch.var(torch.sign(output_stage2[labels_stage2 == 1]).float()).item()

        self.model.train()

        if self.source_exists:
            type2_error_rate_source = self._calculate_type2_error_rate(output_stage2[labels_stage2 == 2])
            return (type1_error_rate, type2_error_rate_target, type2_error_rate_source), target_abnormal_output_variance

        return (type1_error_rate, type2_error_rate_target), target_abnormal_output_variance

    def _evaluate_on_test_data(self, best_point):
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
                    output_test = output_test - self.optimal_threshold # Subtract the threshold
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

        self.model.train()

        # Reset initial states
        self._restore_initial_states()

        return test_error_dict

    ###########################################################################

    def _calculate_optimal_threshold(self, output_stage2, labels_stage2):
        # Calculate the optimal threshold to meet the Type I Error Rate constraints.
        threshold_min = output_stage2[labels_stage2 == 0].min()
        threshold_maxes = [output_stage2[labels_stage2 == 0].max(), torch.max(output_stage2[labels_stage2 == 1].max(), output_stage2[labels_stage2 == 0].max())]
        num_thresholds = [1500, 5000, 10000]

        # We try it again with a higher max if none are achievable, in cases where the model performs badly
        for num_threshold in num_thresholds:
            for threshold_max in threshold_maxes:
                thresholds = torch.linspace(threshold_min, threshold_max, num_threshold)
                best_threshold = None
                best_type1_error_rate = float('-inf')  # Start with the worst possible value
                alpha = (self.type1_error_lowerbound + self.type1_error_upperbound)/2

                for threshold in thresholds:
                    thresholded_output_stage2 = output_stage2 - threshold
                    type1_error_rate = self._calculate_type1_error_rate(thresholded_output_stage2[labels_stage2 == 0])

                    if self.type1_error_lowerbound <= type1_error_rate <= self.type1_error_upperbound:
                        # Check if this threshold is closer to alpha
                        if abs(type1_error_rate-alpha) < abs(best_type1_error_rate-alpha):
                            best_type1_error_rate = type1_error_rate
                            best_threshold = threshold.item()
                if best_threshold:
                    break

        return best_threshold if best_threshold is not None else thresholds[0].item()
import torch
import json
import time

from tlnp_lib.transfer_learning_neyman_pearson import TransferLearningNeymanPearson

class NaiveNeymanPearson(TransferLearningNeymanPearson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_results['approach_name'] = 'nnp'

    def run_training_process(self):
        start_time = time.time()
        lambda_source = 1.0
        lambda_normal = 1.0
        self.train_one_lambda_pair(lambda_source, lambda_normal)

        test_error_dict = self._evaluate_on_test_data((lambda_source, lambda_normal), model_suffix="_nnp")
        self.all_results["test_metrics"] = {
            **test_error_dict,
            'num_trainings': len(self.all_results['training_results'])
        }

        process_time = time.time() - start_time
        self.all_results['process_time'] = process_time
        
        if self.results_save_path:
            with open(self.results_save_path + '_nnp.json', 'w') as f:
                json.dump(self.all_results, f, indent=4)

        return self.all_results

    ###########################################################################

    def _measure_evaluation_metrics(self, output_evaluation):
        # Find threshold to meet Type I Error constraints
        self.optimal_threshold = self._calculate_optimal_threshold(output_evaluation)
        self.all_results["optimal_threshold"] = self.optimal_threshold

        # Adjust outputs based on threshold
        output_evaluation = output_evaluation - self.optimal_threshold

        # Calculate evaluation error rates
        type1_error_rate = self.utils.calculate_type1_error_rate(output_evaluation[self.labels_evaluation == 0])
        type2_error_rate_target = self.utils.calculate_type2_error_rate(output_evaluation[self.labels_evaluation == 1])
        if self.source_exists:
            type2_error_rate_source = self.utils.calculate_type2_error_rate(output_evaluation[self.labels_evaluation == 2])
            return (type1_error_rate, type2_error_rate_target, type2_error_rate_source)

        return (type1_error_rate, type2_error_rate_target)

    def _calculate_optimal_threshold(self, output_evaluation):
        # Calculate the optimal threshold to meet the Type I Error Rate constraints.
        threshold_min = output_evaluation[self.labels_evaluation == 0].min()
        threshold_maxes = [output_evaluation[self.labels_evaluation == 0].max(), torch.max(output_evaluation[self.labels_evaluation == 1].max(), output_evaluation[self.labels_evaluation == 0].max())]
        num_thresholds = [1500, 5000, 10000]

        # We try it again with a higher max if none are achievable, in cases where the model performs badly
        for num_threshold in num_thresholds:
            for threshold_max in threshold_maxes:
                thresholds = torch.linspace(threshold_min, threshold_max, num_threshold)
                best_threshold = None
                best_type1_error_rate = float('-inf')  # Start with the worst possible value
                alpha = (self.type1_error_lowerbound + self.type1_error_upperbound)/2

                for threshold in thresholds:
                    thresholded_output_evaluation = output_evaluation - threshold
                    type1_error_rate = self.utils.calculate_type1_error_rate(thresholded_output_evaluation[self.labels_evaluation == 0])

                    if self.type1_error_lowerbound <= type1_error_rate <= self.type1_error_upperbound:
                        # Check if this threshold is closer to alpha
                        if abs(type1_error_rate-alpha) < abs(best_type1_error_rate-alpha):
                            best_type1_error_rate = type1_error_rate
                            best_threshold = threshold.item()
                if best_threshold:
                    break

        return best_threshold if best_threshold is not None else thresholds[0].item()

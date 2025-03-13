import math

class PointSelectionUtils:
    
    @staticmethod
    def find_point_with_lowest_type2_error_source(filtered_points):
        return min(
            filtered_points,
            key=lambda p: (
                p['evaluation_metrics']['type2_error_rate_source'],
                p['evaluation_metrics']['type2_error_rate_target'],
                p['evaluation_metrics']['type1_error_rate']
            )
        )

    @staticmethod
    def choose_final_point_constant_method(filtered_points, logger, selection_constant, data_dict):
        # Sort and choose best point based on type2_error_rate_target
        best_point = min(filtered_points, key=lambda p: p['evaluation_metrics']['type2_error_rate_target'])
        logger.log_point_selection(f"Best point by Type-II Error (Target):")
        logger.log_point_selection_with_metrics(best_point)

        # Calculate Type II error upperbound for the target
        target_type2_upperbound = best_point['evaluation_metrics']['type2_error_rate_target'] + (selection_constant * (1 / math.sqrt(data_dict["target_abnormal_data"].size(0))))

        # Further filter points based on Type-II error rate (target)
        final_filtered_points = [p for p in filtered_points if p['evaluation_metrics']['type2_error_rate_target'] <= target_type2_upperbound]
        num_points_threshold2 = len(final_filtered_points)
        logger.log_point_selection(f"Points below Type-II (Target) threshold of {round(target_type2_upperbound,4)}: {num_points_threshold2}")

        # Choose the point with the lowest type2_error_rate_source
        final_point = PointSelectionUtils.find_point_with_lowest_type2_error_source(final_filtered_points)
        logger.log_point_selection(f"Final point:")
        logger.log_point_selection_with_metrics(final_point)

        return final_point, num_points_threshold2, target_type2_upperbound

    @staticmethod
    def filter_points_by_type1_error(all_results, error_threshold):
        filtered_points = []
        for key, result in all_results["training_results"].items():
            try:
                if result['evaluation_metrics']['type1_error_rate'] <= error_threshold:
                    filtered_points.append(result)
            except:
                pass
        return filtered_points

    @staticmethod
    def find_best_point_after_failures(all_results, type1_error_upperbound):
        # Filter all results to include only points with Type I error below the upper bound and that have 'evaluation_metrics'
        eligible_points = [(key, value) for key, value in all_results["training_results"].items()
                          if isinstance(value, dict) and 'evaluation_metrics' in value and value['evaluation_metrics']['type1_error_rate'] < type1_error_upperbound]

        if not eligible_points:
            # If no points meet the Type I error threshold, consider all points that have 'evaluation_metrics'
            eligible_points = [(key, value) for key, value in all_results["training_results"].items()
                              if isinstance(value, dict) and 'evaluation_metrics' in value]

        # If still no eligible points found, raise an error
        if not eligible_points:
            return None

        # Sort points by Type II error rate, then Type I error rate, then order of training
        eligible_points.sort(key=lambda x: (
            x[1]['evaluation_metrics']['type2_error_rate_target'],
            x[1]['evaluation_metrics']['type1_error_rate']
        ))

        # Return the best point based on sorting criteria
        best_point_key, best_point_value = eligible_points[0]
        return (best_point_value['lambda_source'], best_point_value['lambda_normal'],
                best_point_value['evaluation_metrics']['type1_error_rate'],
                best_point_value['evaluation_metrics']['type2_error_rate_target'])

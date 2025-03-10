import json
import math
import os
import time
from glob import glob
import pandas as pd

class ExperimentationUtils:
    
    def read_and_combine_json_files(path_pattern, num_files=10, type1_upperbound=None):
        file_paths = glob(path_pattern)
        method1_constant=1.5
        print(f"Number of files found: {len(file_paths)}")
        if len(file_paths) > num_files:
            file_paths = file_paths[:num_files]
            print(f"Took first {num_files} files")
        all_rows = []

        for file_path in file_paths:
            with open(file_path, 'r') as f:
                model_output = json.load(f)

            # Get the abnormal data size
            abnormal_data_size = _get_abnormal_data_size(model_output)

            for key, results in model_output.items():
                if key == 'config' or results == {}:
                    continue

                if key == "only_target_neyman_pearson":
                    source_points = 'No source'
                    approach_type = 'Only Target Neyman-Pearson'
                    append_row_to_list(all_rows, results['test_metrics']['best_lambda_source'], results['test_metrics']['best_lambda_normal'], results['test_metrics']['type1_error_test'], results['test_metrics']['type2_error_test'], results['test_metrics']['num_trainings'], approach_type, source_points)

                elif key == "only_target_thresholding_traditional_classification":
                    source_points = 'No source'
                    approach_type = 'Only Target Thresholding Traditional Classification'
                    append_row_to_list(all_rows, None, None, results['test_metrics']['type1_error_test'], results['test_metrics']['type2_error_test'], results['test_metrics']['num_trainings'], approach_type, source_points)

                # Handling the key 'transfer_learning_neyman_pearson'
                if key.startswith("source_points_"):
                    source_points = float(key.split('_')[-1])

                    for key1, results1 in results.items():
                        # Process Method 1 and Method 2 for 'transfer_learning_neyman_pearson'
                        if key1 == 'transfer_learning_neyman_pearson':
                            num_trainings = results1["num_trainings"]
                            method1 = results1['test_metrics_method1']

                            # Process Method 1
                            method1_filtered_points = _filter_points_by_type1_error(
                                results1, type1_upperbound if type1_upperbound else method1['type1_upperbound']
                            )
                            best_point = _find_best_point(method1_filtered_points)
                            method1_upperbound = _calculate_target_type2_upperbound_method1(best_point, method1_constant, abnormal_data_size)
                            final_filtered_points = _apply_second_filter(method1_filtered_points, method1_upperbound)
                            final_point_method1 = _find_point_with_lowest_type2_error_source(final_filtered_points)

                            # Append the final row for Method 1
                            append_row_to_list(all_rows, final_point_method1['lambda_source'], final_point_method1['lambda_normal'],
                                                final_point_method1['test_metrics']['type1_error_test'], final_point_method1['test_metrics']['type2_error_test'],
                                                num_trainings,
                                                'Transfer Learning Neyman-Pearson', source_points)

                        elif key1 == "pooled_source_and_target_neyman_pearson":
                            approach_type = 'Pooled Source and Target Neyman-Pearson'
                            append_row_to_list(all_rows, results1['test_metrics']['best_lambda_source'], results1['test_metrics']['best_lambda_normal'], results1['test_metrics']['type1_error_test'], results1['test_metrics']['type2_error_test'], results1['test_metrics']['num_trainings'], approach_type, source_points)

                        elif key1 == "only_source_neyman_pearson":
                            approach_type = 'Only Source Neyman-Pearson'
                            append_row_to_list(all_rows, results1['test_metrics']['best_lambda_source'], results1['test_metrics']['best_lambda_normal'], results1['test_metrics']['type1_error_test'], results1['test_metrics']['type2_error_test'], results1['test_metrics']['num_trainings'], approach_type, source_points)

                        elif key1 == "pooled_source_and_target_thresholding_traditional_classification":
                            approach_type = 'Pooled Source and Target Thresholding Traditional Classification'
                            append_row_to_list(all_rows, None, None, results1['test_metrics']['type1_error_test'], results1['test_metrics']['type2_error_test'], results1['test_metrics']['num_trainings'], approach_type, source_points)

                        elif key1 == "transfer_learning_outlier_detection":
                            approach_type = 'Transfer Learning Outlier Detection'
                            append_row_to_list(all_rows, results1['test_metrics']['best_lambda_source'], results1['test_metrics']['best_lambda_normal'], results1['test_metrics']['type1_error_test'], results1['test_metrics']['type2_error_test'], results1['test_metrics']['num_trainings'], approach_type, source_points)

                        else:
                            continue
                else:
                    continue

        df = pd.DataFrame(all_rows)
        return df

    def _get_abnormal_data_size(model_output):
        # This function gets the abnormal data size from multiple config sources.
        config = model_output['config']

        # Check for 'num_target_abnormal_training' in 'climsim_config' and 'nasa_config'
        for key in ['climsim_config', 'nasa_config', 'credit_config', 'plantvillage_config']:
            if key in config and 'num_target_abnormal_training' in config[key]:
                return config[key]['num_target_abnormal_training']

        # Check for 'num_training_datapoints' in 'target_abnormal_class_config' inside 'data_generation_config'
        if 'data_generation_config' in config and 'target_abnormal_class_config' in config['data_generation_config']:
            if 'num_training_datapoints' in config['data_generation_config']['target_abnormal_class_config']:
                return config['data_generation_config']['target_abnormal_class_config']['num_training_datapoints']

        # Raise an error if none of the conditions are met
        raise ValueError("Could not find the abnormal data size in the provided config")

    def _filter_points_by_type1_error(results, error_threshold):
        filtered_points = []
        for key, result in results.items():
            if key.startswith("lambda_source") and 'stage2_metrics' in result:
                if result['stage2_metrics']['type1_error_rate'] <= error_threshold:
                    filtered_points.append(result)
        return filtered_points

    def _find_best_point(filtered_points):
        # Find the point with the lowest type2_error_rate_target
        return min(filtered_points, key=lambda p: p['stage2_metrics']['type2_error_rate_target'])

    def _calculate_target_type2_upperbound_method1(best_point, method1_constant, abnormal_data_size):
        return best_point['stage2_metrics']['type2_error_rate_target'] + (method1_constant * (1 / math.sqrt(abnormal_data_size)))

    def _apply_second_filter(filtered_points, upperbound):
        # Filter points by the calculated target type2 upperbound
        return [p for p in filtered_points if p['stage2_metrics']['type2_error_rate_target'] <= upperbound]

    def _find_point_with_lowest_type2_error_source(filtered_points):
        # Choose the point with the lowest type2_error_rate_source, breaking ties with target error rates
        return min(filtered_points, key=lambda p: (
            p['stage2_metrics']['type2_error_rate_source'],
            p['stage2_metrics']['type2_error_rate_target'],
            p['stage2_metrics']['type1_error_rate']
        ))

    # Main function to execute the process
    def process_experiment_results(path_pattern, num_files, graph_title, type1_upperbound=None):
        df = read_and_combine_json_files(path_pattern, num_files, type1_upperbound)

        # Process the DataFrame to calculate averages
        averaged_df = average_results(df)

        # Move 'Source Points' and 'Approach Type' columns to the first and second positions
        cols = ['Source Points', 'Approach Type'] + [col for col in averaged_df.columns if col not in ['Source Points', 'Approach Type']]
        averaged_df = averaged_df[cols]

        # Plot the graphs
        plot_graphs(averaged_df, graph_title)

        # Display the table using tabulate
        print(tabulate(averaged_df, headers='keys', tablefmt='github'))
        
    def run_experiment_iteration(i, seed, model, optimizer, scheduler, loss_function, training_config, debug_modes,
                                target_normal_train, target_normal_test, target_abnormal_train, target_abnormal_test,
                                source_abnormal_train, num_source_points_list, config_path, config):

        print_experiment_start(i, seed)
        divider = "─" * 100
        model_output = {
            'config': config,
            'seed': seed
        }
        experiment_start_time = time.time()

        # Only target Neyman-Pearson
        print(f"\n{divider}\nRunning Only Target Neyman-Pearson...\n{divider}\n")
        data_dict = {
            'target_normal_data': target_normal_train,
            'target_abnormal_data': target_abnormal_train,
            'target_normal_test_data': target_normal_test,
            'target_abnormal_test_data': target_abnormal_test,
            'secondary_abnormal_test_data': target_abnormal_train
        }
        tlnp_trainer = TransferLearningNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                    data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
        results = tlnp_trainer.run_training_process()
        model_output["only_target_neyman_pearson"] = results


        # Only Target Thresholding Traditional Classification
        print(f"\n\n{divider}\nRunning Only Target Thresholding Traditional Classification...\n{divider}\n")
        data_dict = {
            'target_normal_data': target_normal_train,
            'target_abnormal_data': target_abnormal_train,
            'target_normal_test_data': target_normal_test,
            'target_abnormal_test_data': target_abnormal_test
        }
        tlnp_trainer = NaiveNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                    data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
        results = tlnp_trainer.run_training_process()
        model_output["only_target_thresholding_traditional_classification"] = results

        # Running following approaches separately for each subset of source points
        for num_source_points, i in zip(num_source_points_list, range(len(num_source_points_list))):
            print_source_points(num_source_points)
            source_abnormal_train_subset = source_abnormal_train[:num_source_points]
            source_points_dict = {}

            # Transfer Learning Neyman Pearson
            print(f"\n{divider}\nRunning Transfer Learning Neyman Pearson, with {num_source_points} source points...\n{divider}\n")
            data_dict = {
                'target_normal_data': target_normal_train,
                'target_abnormal_data': target_abnormal_train,
                'target_normal_test_data': target_normal_test,
                'target_abnormal_test_data': target_abnormal_test,
                'source_abnormal_data': source_abnormal_train_subset
            }
            tlnp_trainer = TransferLearningNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                        data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
            results = tlnp_trainer.run_training_process()
            source_points_dict["transfer_learning_neyman_pearson"] = results

            # Pooled Source and Target Neyman Pearson
            print(f"\n\n{divider}\nRunning Pooled Source and Target Neyman Pearson, with {num_source_points} source points...\n{divider}\n")
            # Combine source_abnormal_train with target_abnormal_train
            combined_abnormal_train = torch.cat((target_abnormal_train, source_abnormal_train_subset), dim=0)
            data_dict = {
                'target_normal_data': target_normal_train,
                'target_abnormal_data': combined_abnormal_train,
                'target_normal_test_data': target_normal_test,
                'target_abnormal_test_data': target_abnormal_test,
            }
            tlnp_trainer = TransferLearningNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                        data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
            results = tlnp_trainer.run_training_process()
            source_points_dict["pooled_source_and_target_neyman_pearson"] = results

            # Only Source Neyman Pearson
            print(f"\n\n{divider}\nRunning Only Source Neyman Pearson, with {num_source_points} source points...\n{divider}\n")
            data_dict = {
                'target_normal_data': target_normal_train,
                'target_abnormal_data': source_abnormal_train_subset,
                'target_normal_test_data': target_normal_test,
                'target_abnormal_test_data': target_abnormal_test,
                'secondary_abnormal_test_data': target_abnormal_train
            }
            tlnp_trainer = TransferLearningNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                        data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
            results = tlnp_trainer.run_training_process()
            source_points_dict["only_source_neyman_pearson"] = results

            # Pooled source and target thresholding traditional classification
            print(f"\n\n{divider}\nRunning Pooled Source and Target Thresholding Traditional Classification, with {num_source_points} source points...\n{divider}\n")
            data_dict = {
                'target_normal_data': target_normal_train,
                'target_abnormal_data': target_abnormal_train,
                'target_normal_test_data': target_normal_test,
                'target_abnormal_test_data': target_abnormal_test,
                'source_abnormal_data': source_abnormal_train_subset
            }
            tlnp_trainer = NaiveNeymanPearson(model=model, optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                                        data_dict = data_dict, config = training_config, seed = seed, debug_modes=debug_modes)
            results = tlnp_trainer.run_training_process()
            source_points_dict["pooled_source_and_target_thresholding_traditional_classification"] = results

            # Transfer Learning Outlier Detection
            only_target_type2_error = model_output["only_target_neyman_pearson"]["test_metrics"]["secondary_type2_error_test"]
            only_source_type2_error = source_points_dict["only_source_neyman_pearson"]["test_metrics"]["secondary_type2_error_test"]
            results = {}
            if only_target_type2_error < only_source_type2_error:
                results["test_metrics"] = model_output["only_target_neyman_pearson"]["test_metrics"]
                results["chosen_approach"] = "only_target_neyman_pearson"
            else:
                results["test_metrics"] = source_points_dict["only_source_neyman_pearson"]["test_metrics"]
                results["chosen_approach"] = "only_source_neyman_pearson"
            key = "transfer_learning_outlier_detection"
            source_points_dict[key] = results
            print(f"\n\n{divider}\nTransfer Learning Outlier Detection: Selected approach is {source_points_dict[key]['chosen_approach']}\n{divider}\n")

            # Store source_points_dict in model_output_dict
            key = f"source_points_{num_source_points}"
            model_output[key] = source_points_dict

        save_results(config_path, model_output)
        print_experiment_time(experiment_start_time)
        
    def plot_graphs(results_df, graph_title = ''):
        fig, ax = plt.subplots(figsize=(10, 6))
        color_cycler = plt.cm.tab20(np.linspace(0, 1, 20))  # Use tab20 with 20 distinct colors
        ax.set_prop_cycle('color', color_cycler)
        i = 0

        # Plot method1
        approach = "Transfer Learning Neyman-Pearson"
        approach_df = results_df[(results_df['Approach Type'] == approach) & (results_df['Source Points'] != 'No source')].sort_values(by='Source Points')
        if not approach_df.empty:
            ax.plot(approach_df['Source Points'], approach_df['Test Type2 Error'], marker='o', label=f'{approach}', linewidth=2.5, color='r', markersize = 4)

        # Plot approaches with source
        for approach in ['Pooled Source and Target Neyman-Pearson', 'Only Source Neyman-Pearson', 'Pooled Source and Target Thresholding Traditional Classification', 'Transfer Learning Outlier Detection']:
            approach_df = results_df[(results_df['Approach Type'] == approach) & (results_df['Source Points'] != 'No source')].sort_values(by='Source Points')
            if not approach_df.empty:
                ax.plot(approach_df['Source Points'], approach_df['Test Type2 Error'], marker='o', label=f'{approach}', color = color_cycler[i], markersize=4)
                i += 1

        # Add horizontal lines for approaches with only target
        for approach, linestyle in zip(['Only Target Neyman-Pearson', 'Only Target Thresholding Traditional Classification'], ['--',':']):
            approach_df = results_df[(results_df['Approach Type'] == approach)]
            if not approach_df.empty:
                ax.axhline(y=approach_df['Test Type2 Error'].values[0], linestyle=linestyle, label=f'{approach}', color = color_cycler[i])
                i += 1

        ax.set_xlabel('Source Points')
        ax.set_ylabel('Test Type2 Error')
        ax.set_title(graph_title)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def average_results(df):
        result_rows = []

        grouped = df.groupby(['Source Points', 'Approach Type'])
        for (source_points, approach_type), group in grouped:
            avg_row = group.mean(numeric_only=True)
            avg_row['Source Points'] = source_points
            avg_row['Approach Type'] = approach_type
            result_rows.append(avg_row)

        result_df = pd.DataFrame(result_rows)
        return result_df

    def append_row_to_list(all_rows, lambda_source, lambda_normal, type1_error_test, type2_error_test, num_trainings, approach_type, source_points):
        row = {
            'Source Points': source_points,
            'Approach Type': approach_type,
            'Best Lambda Source': lambda_source,
            'Best Lambda Normal': lambda_normal,
            'Test Type1 Error': type1_error_test,
            'Test Type2 Error': type2_error_test,
            'Num Trainings': num_trainings
        }
        all_rows.append(row)
        return all_rows
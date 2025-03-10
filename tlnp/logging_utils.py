import os
import json
import time

class LoggingUtils:

    @staticmethod
    def save_results(config_path, model_output):
        # Generate the base file name
        base_file_name = f"{os.path.splitext(os.path.basename(config_path))[0]}_training_results"
        results_dir = 'results'
        results_file_name = f"{base_file_name}.json"
        results_path = os.path.join(results_dir, results_file_name)

        # Use seed if it exists
        if "seed" in model_output:
            results_file_name = f"{base_file_name}_{model_output['seed']}.json"
            results_path = os.path.join(results_dir, results_file_name)

        # Otherwise, use numeric suffix
        else:
            counter = 1
            while os.path.exists(results_path):
                results_file_name = f"{base_file_name}_{counter}.json"
                results_path = os.path.join(results_dir, results_file_name)
                counter += 1

        # Save the results
        with open(results_path, 'w') as f:
            json.dump(model_output, f, indent=4)

        LoggingUtils.print_data_saved_message(results_path)

    @staticmethod
    def print_experiment_start(i, seed, line_length=100):
        top_border = '╔' + '═' * (line_length - 2) + '╗'
        bottom_border = '╚' + '═' * (line_length - 2) + '╝'
        side_border = '║' + ' ' * (line_length - 2) + '║'
        message = f"STARTING EXPERIMENT ITERATION #{i+1} WITH SEED OF {seed}"
        print(f"\n{top_border}\n{side_border}\n║{message.center(line_length-2)}║\n{side_border}\n{bottom_border}\n")

    @staticmethod
    def print_source_points(source_points, line_length=100):
        top_border = '┌' + '─' * (line_length - 2) + '┐'
        bottom_border = '└' + '─' * (line_length - 2) + '┘'
        side_border = '│' + ' ' * (line_length - 2) + '│'
        message = f"RUNNING APPROACHES USING {source_points} SOURCE POINTS"
        print(f"\n{top_border}\n{side_border}\n│{message.center(line_length-2)}│\n{side_border}\n{bottom_border}")

    @staticmethod
    def print_experiment_time(experiment_start_time, line_length=100):
        # Calculate elapsed time
        elapsed_time = time.time() - experiment_start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        # Format the message
        if hours > 0:
            message = f"Experiment iteration took {hours} hours and {minutes} minutes to complete."
        else:
            message = f"Experiment iteration took {minutes} minutes to complete."

        # Create the box
        top_border = '┌' + '─' * (line_length - 2) + '┐'
        bottom_border = '└' + '─' * (line_length - 2) + '┘'

        # Print with a box around the message
        print(f"\n{top_border}\n│{message.center(line_length-2)}│\n{bottom_border}")

    @staticmethod
    def print_data_saved_message(results_path, line_length=100):
        # Format the message
        message = f"Data saved to {results_path}"

        # Create the box
        top_border = '┌' + '─' * (line_length - 2) + '┐'
        bottom_border = '└' + '─' * (line_length - 2) + '┘'

        # Print with a box around the message (without extra side borders)
        print(f"\n{top_border}\n│{message.center(line_length-2)}│\n{bottom_border}")

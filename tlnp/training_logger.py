import logging
import matplotlib.pyplot as plt

class TrainingLogger:
    def __init__(self, debug_config):
        """Initialize the logger with a dictionary of debug modes."""
        self.debug_config = debug_config
        self.log_filename = debug_config.get("log_filename", None)
        self.debug_config['main_logger'] = True  # Always enable the main logger

        class DebugFilter(logging.Filter):
            """Allow logging only if the corresponding debug mode is enabled."""
            def filter(self, record):
                return debug_config.get(record.name, False)

        # Define loggers for each debug mode
        self.loggers = {
            'main_logger': logging.getLogger('main_logger'),
            'training_progress': logging.getLogger('print_training_progress'),
            'lr_changes': logging.getLogger('print_lr_changes'),
            'lambda_updates': logging.getLogger('print_lambda_updates'),
            'point_selection': logging.getLogger('print_point_selection')
        }

        # Configure each logger
        for name, logger in self.loggers.items():
            logger.setLevel(logging.INFO)
            logger.addFilter(DebugFilter())
            
            # Remove existing handlers before adding new ones
            if logger.hasHandlers():
                logger.handlers.clear()
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler (optional)
            if self.log_filename:
                file_handler = logging.FileHandler(self.log_filename, mode='a')  # Append mode
                file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

    # Logging methods for each debug mode
    def log(self, message):
        self.loggers['main_logger'].info(message)

    def log_training_progress(self, message):
        if self.debug_config.get('print_training_progress', False):
            self.loggers['training_progress'].info(message)

    def log_training_progress_losses(self, epoch, loss, val_loss, total_epoch_time, num_epochs):
        if self.debug_config.get('print_training_progress', False) and epoch % self.debug_config.get('print_training_epoch_frequency', 25) == 0:
            average_epoch_time = total_epoch_time / (epoch + 1)  # Add +1 because epoch is zero-indexed
            self.loggers['training_progress'].info(f"Epoch {epoch}/{num_epochs}, Training Loss: {loss:.6f}, Validation Loss: {val_loss:.6f}, "
                f"Avg Epoch Time: {average_epoch_time:.2f} seconds")

    def log_lr_changes(self, message):
        if self.debug_config.get('print_lr_changes', False):
            self.loggers['lr_changes'].info(message)

    def log_lambda_updates(self, message):
        if self.debug_config.get('print_lambda_updates', False):
            self.loggers['lambda_updates'].info(message)

    def log_point_selection(self, message):
        if self.debug_config.get('print_point_selection', False):
            self.loggers['point_selection'].info(message)

    def log_point_selection_with_metrics(self, point):
        if self.debug_config.get('print_point_selection', False):
            message = f"      Lambda Source: {point['lambda_source']}\n      Lambda Normal: {point['lambda_normal']}\n"
            message += f"      Evaluation Type-I Error: {point['evaluation_metrics']['type1_error_rate']}\n"
            message += f"      Evaluation Type-II Error (Target): {point['evaluation_metrics']['type2_error_rate_target']}\n"
            message += f"      Evaluation Type-II Error (Source): {point['evaluation_metrics']['type2_error_rate_source']}"
            self.loggers['point_selection'].info(message)

    def show_training_loss_plot(self, epoch_training_losses, epoch_validation_losses, lr_change_epochs):
        """Displays loss plots if the corresponding debug mode is enabled."""
        if self.debug_config.get('show_losses_plots', False):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.plot(epoch_training_losses)
            ax1.set_title('Training Loss')

            if self.debug_config.get('show_lr_changes', False):
                for epoch in lr_change_epochs:
                    ax1.axvline(x=epoch, color='r', linestyle='--', label=f'LR reduced at epoch {epoch}')

            # Apply log scale if specified in debug_modes
            if self.debug_config.get('log_scale', False):
                ax1.set_yscale('log')
                ax2.set_yscale('log')

            ax2.plot(epoch_validation_losses)
            ax2.set_title('Validation Loss')
            plt.show()

from tlnp.naive_neyman_pearson import NaiveNeymanPearson
from tlnp.optimizers import Optimizers
from tlnp.transfer_learning_neyman_pearson import TransferLearningNeymanPearson
from tlnp.loss import LossFunctions
import yaml
import torch.optim as optim

class TLNP:
    
    def run_tlnp(config_path, data_dict, model, loss_function = None, optimizer = None, scheduler = None):
        config, loss_function, optimizer, scheduler = TLNP._process_config_file(config_path, model, loss_function, optimizer, scheduler)
        TLNP._check_data_dict(data_dict)
        
        tlnp = TransferLearningNeymanPearson(config, data_dict, model, loss_function, optimizer, scheduler)
        return tlnp.run_training_process()

    def run_naive_np(config_path, data_dict, model, loss_function = None, optimizer = None, scheduler = None):
        config, loss_function, optimizer, scheduler = TLNP._process_config_file(config_path, model, loss_function, optimizer, scheduler)
        TLNP._check_data_dict(data_dict)
        
        nnp = NaiveNeymanPearson(config, data_dict, model, loss_function, optimizer, scheduler)
        return nnp.run_training_process()
    
    def _process_config_file(config_path, model, loss_function = None, optimizer = None, scheduler = None):
        # Load YAML config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if loss_function is None:
            # Extract loss function from config
            if "loss_function_config" in config:
                loss_functions_instance = LossFunctions(config["loss_function_config"])
                loss_function = loss_functions_instance.loss_function
            else:
                raise ValueError("No loss function provided and no valid loss function config found in the config file.")
        else:
            # Validate the user-provided loss function
            LossFunctions.validate_loss_function(loss_function)
        
        if optimizer is None:
            # Extract optimizer from config
            if "optimizer_config" in config:
                optimizer = Optimizers.get_optimizer(config["optimizer_config"], model.parameters())
            else:
                raise ValueError("No optimizer provided and no valid optimizer config found in the config file.")
        else:
            # Validate that the provided optimizer is a valid PyTorch optimizer
            if not isinstance(optimizer, optim.Optimizer):
                raise TypeError(f"Invalid optimizer type: {type(optimizer)}. Expected an instance of torch.optim.Optimizer.")

        if scheduler is None:
            # Extract scheduler from config if provided
            if "scheduler_config" in config:
                scheduler = Optimizers.get_scheduler(config["scheduler_config"], optimizer)
        else:
            # Validate that the provided optimizer is a valid PyTorch scheduler
            if not isinstance(optimizer, optim.lr_scheduler._LRScheduler):
                raise TypeError(f"Invalid optimizer type: {type(optimizer)}. Expected an instance of torch.optim.Optimizer.")

        return config, loss_function, optimizer, scheduler

    def _check_data_dict(data_dict):
        # Check if there are missing keys
        if 'target_abnormal_data' not in data_dict or 'target_normal_data' not in data_dict:
            raise ValueError("data_dict must contain 'target_abnormal_data' and 'target_normal_data'.")

        # Get the number of columns for the first dataset in the dictionary
        first_key = next(iter(data_dict))
        num_columns = data_dict[first_key].size(1)

        # Iterate over the datasets and check if they have the same number of columns
        for key, tensor in data_dict.items():
            if tensor.size(1) != num_columns:
                raise ValueError(f"Dataset '{key}' has {tensor.size(1)} columns, expected {num_columns} columns.")

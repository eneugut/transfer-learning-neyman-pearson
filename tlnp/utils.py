import torch.optim as optim

from tlnp.models.quadratic_form_model import QuadraticFormModel
from tlnp.models.mlp_model import MultiLayerPerceptron
from tlnp.loss import LossFunctions

class Utils:
    @staticmethod
    def get_model(config):
        model_name = config.get('model_name', None)
        input_dim = config.get('input_dim', 0)
        hidden_dim = config.get('hidden_dim', input_dim//2)

        if model_name == "QuadraticForm":
            return QuadraticFormModel(input_dim)
        elif model_name == "MultiLayerPerceptron":
            return MultiLayerPerceptron(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    @staticmethod
    def get_loss_function(loss_function_type):
        if loss_function_type == "ExponentialLoss":
            return LossFunctions.exponential_loss_function
        elif loss_function_type == "LogisticLoss":
            return LossFunctions.logistic_loss_function
        elif loss_function_type == "HingeLoss":
            return LossFunctions.hinge_loss_function
        else:
            raise ValueError(f"Unknown loss function: {loss_function_type}")

    @staticmethod
    def get_optimizer(optimizer_config, model_parameters):
        optimizer_name = optimizer_config['optimizer_name']
        optimizer_params = optimizer_config.get(optimizer_name, {})
        learning_rate = optimizer_params.get('learning_rate', 0.001)

        if optimizer_name == "SGD":
            return optim.SGD(model_parameters, lr=learning_rate, momentum=optimizer_params.get('momentum', 0.0))
        elif optimizer_name == "Adam":
            return optim.Adam(model_parameters, lr=learning_rate, betas=optimizer_params.get('betas', (0.9, 0.999)))
        elif optimizer_name == "RMSprop":
            return optim.RMSprop(model_parameters, lr=learning_rate, alpha=optimizer_params.get('alpha', 0.99))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    @staticmethod
    def get_scheduler(scheduler_config, optimizer):
        scheduler_name = scheduler_config['scheduler_name']
        scheduler_params = scheduler_config.get(scheduler_name, {})

        if scheduler_name == None:
            return None
        elif scheduler_name == "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params.get('step_size', 30), gamma=scheduler_params.get('gamma', 0.1))
        elif scheduler_name == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params.get('gamma', 0.9))
        elif scheduler_name == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.get('mode', 'min'),
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 10),
                threshold=scheduler_params.get('threshold', 0.0001),
                threshold_mode=scheduler_params.get('threshold_mode', 'rel'),
                cooldown=scheduler_params.get('cooldown', 0),
                min_lr=scheduler_params.get('min_lr', 0),
                eps=scheduler_params.get('eps', 1e-08)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_name}")

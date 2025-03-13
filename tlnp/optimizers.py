import torch.optim as optim

class Optimizers:

    @staticmethod
    def get_optimizer(optimizer_config, model_parameters):
        optimizer_type = optimizer_config.get('optimizer_type', None)
        optimizer_params = optimizer_config.get(optimizer_type, {})
        learning_rate = optimizer_params.get('learning_rate', 0.001)

        if optimizer_type == "SGD":
            return optim.SGD(model_parameters, lr=learning_rate, momentum=optimizer_params.get('momentum', 0.0))
        elif optimizer_type == "Adam":
            return optim.Adam(model_parameters, lr=learning_rate, betas=optimizer_params.get('betas', (0.9, 0.999)))
        elif optimizer_type == "RMSprop":
            return optim.RMSprop(model_parameters, lr=learning_rate, alpha=optimizer_params.get('alpha', 0.99))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    @staticmethod
    def get_scheduler(scheduler_config, optimizer):
        scheduler_type = scheduler_config.get('scheduler_type', None)
        scheduler_params = scheduler_config.get(scheduler_type, {})

        if scheduler_type == None:
            return None
        elif scheduler_type == "StepLR":
            return optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params.get('step_size', 30), gamma=scheduler_params.get('gamma', 0.1))
        elif scheduler_type == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params.get('gamma', 0.9))
        elif scheduler_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 10),
                threshold=scheduler_params.get('threshold', 0.0001),
                threshold_mode=scheduler_params.get('threshold_mode', 'rel'),
                cooldown=scheduler_params.get('cooldown', 0),
                min_lr=scheduler_params.get('min_lr', 0),
                eps=scheduler_params.get('eps', 1e-08)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

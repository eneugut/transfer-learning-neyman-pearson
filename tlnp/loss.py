import torch
import inspect

class LossFunctions:
    def __init__(self, loss_config):
        loss_function_type = loss_config.get("loss_function_type", None)
        self.normalize_losses = loss_config.get("normalize_losses", False)
        self.clip_value = loss_config.get("clip_value", 20)
        if loss_function_type == "ExponentialLoss":
            self.loss_function = self.exponential_loss_function
        elif loss_function_type == "LogisticLoss":
            self.loss_function = self.logistic_loss_function
        elif loss_function_type == "HingeLoss":
            self.loss_function = self.hinge_loss_function
        else:
            raise ValueError(f"Unknown loss function: {loss_function_type}")
    
    def exponential_loss_function(self, y_pred, labels, lambda_target, lambda_source, lambda_normal):
        if y_pred.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        # Ensure y_pred is at least two-dimensional before applying the mask
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)  # Add an extra dimension to make it compatible

        # Initialize the loss to zero
        loss = torch.tensor(0.0, device=y_pred.device)

        # Compute the loss for each label type
        for label_value, lambda_val in [(1, lambda_target), (2, lambda_source), (0, lambda_normal)]:
            mask = (labels == label_value)
            if mask.sum() > 0:
                y_true = torch.ones_like(y_pred[mask]) if label_value in [1, 2] else -torch.ones_like(y_pred[mask])
                exp_term = -y_true * y_pred[mask]
                exp_term = torch.clip(exp_term, max=self.clip_value)
                current_loss = torch.mean(lambda_val * torch.exp(exp_term))

                if self.normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

    def logistic_loss_function(self, y_pred, labels, lambda_target, lambda_source, lambda_normal):
        if y_pred.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        # Initialize the loss to zero
        loss = torch.tensor(0.0, device=y_pred.device)

        # Ensure y_pred is at least two-dimensional before applying the mask
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)  # Add an extra dimension to make it compatible

        # Compute the loss for each label type
        for label_value, lambda_val in [(1, lambda_target), (2, lambda_source), (0, lambda_normal)]:
            mask = (labels == label_value)
            if mask.sum() > 0:
                y_true = torch.ones_like(y_pred[mask]) if label_value in [1, 2] else -torch.ones_like(y_pred[mask])
                exp_term = -y_true * y_pred[mask]
                exp_term = torch.clip(exp_term, max=self.clip_value)
                current_loss = torch.mean(lambda_val * torch.log(1 + torch.exp(exp_term)))

                if self.normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

    def hinge_loss_function(self, y_pred, labels, lambda_target, lambda_source, lambda_normal):
        if y_pred.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        # Ensure y_pred is at least two-dimensional before applying the mask
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)  # Add an extra dimension to make it compatible

        # Initialize the loss to zero
        loss = torch.tensor(0.0, device=y_pred.device)

        # Compute the loss for each label type
        for label_value, lambda_val in [(1, lambda_target), (2, lambda_source), (0, lambda_normal)]:
            mask = (labels == label_value)
            if mask.sum() > 0:
                y_true = torch.ones_like(y_pred[mask]) if label_value in [1, 2] else -torch.ones_like(y_pred[mask])
                hinge_term = torch.clamp(1 - y_true * y_pred[mask], min=0)
                current_loss = torch.mean(lambda_val * hinge_term)

                if self.normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

    @staticmethod
    def validate_loss_function(loss_function):
        """Ensure the loss function has the correct parameters."""
        expected_args = {"y_pred", "labels", "lambda_target", "lambda_source", "lambda_normal"}

        if not callable(loss_function):
            raise ValueError("Provided loss function is not callable.")

        # Get the function signature
        func_signature = inspect.signature(loss_function)
        func_params = set(func_signature.parameters.keys())

        if expected_args.issubset(func_params):
            return True  # Function is valid
        else:
            missing_args = expected_args - func_params
            raise ValueError(f"Loss function is missing required arguments: {missing_args}")

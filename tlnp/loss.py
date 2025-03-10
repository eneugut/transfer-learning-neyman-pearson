import torch

class LossFunctions:
    
    @staticmethod
    def exponential_loss_function(y_pred, labels, lambda_target, lambda_source, lambda_normal, normalize_losses=False, clip_value=20):
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
                exp_term = torch.clip(exp_term, max=clip_value)
                current_loss = torch.mean(lambda_val * torch.exp(exp_term))

                if normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

    @staticmethod
    def logistic_loss_function(y_pred, labels, lambda_target, lambda_source, lambda_normal, normalize_losses=False, clip_value=20):
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
                exp_term = torch.clip(exp_term, max=clip_value)
                current_loss = torch.mean(lambda_val * torch.log(1 + torch.exp(exp_term)))

                if normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

    @staticmethod
    def hinge_loss_function(y_pred, labels, lambda_target, lambda_source, lambda_normal, normalize_losses=False):
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

                if normalize_losses:
                    current_loss /= mask.sum()

                loss += current_loss

        return loss

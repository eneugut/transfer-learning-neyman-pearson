import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxRiskObjective(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        alpha: float,
        epsilon_0_T: float, epsilon_1_T: float,
        epsilon_0_S: float, epsilon_1_S: float,
        R1T_hat_const: float,
        code_T0: int = 0, code_T1: int = 1, code_S1: int = 2, code_S0: int = 3,
        loss_function_type: str = "ExpontentialLoss",      # "LogisticLoss" (BCE), "ExponentialLoss", or "HingeLoss"
    ):
        super().__init__()
        self.model = model
        self.alpha = float(alpha)
        self.epsilon_0_T = float(epsilon_0_T)
        self.epsilon_1_T = float(epsilon_1_T)
        self.epsilon_0_S = float(epsilon_0_S)
        self.epsilon_1_S = float(epsilon_1_S)

        self.code_T0, self.code_T1, self.code_S1, self.code_S0 = int(code_T0), int(code_T1), int(code_S1), int(code_S0)
        self.loss_function_type = loss_function_type

        self.register_buffer("R1T_hat_const", torch.tensor(R1T_hat_const, dtype=torch.float32))
        self.register_buffer("lmbda", torch.tensor(0.0), persistent=False)
        self.alpha_prime = nn.Parameter(torch.tensor(self.alpha, dtype=torch.float32))

    def _logits(self, x):  # [N]
        return self.model(x).squeeze(-1)

    def _split_masks(self, labels):
        lbl = labels.view(-1).long()
        return (lbl == self.code_T0), (lbl == self.code_T1), (lbl == self.code_S0)  # T0, T1, S0

    # --- loss choices ---
    def _loss_on_subset(self, logits_subset: torch.Tensor, target01: float) -> torch.Tensor:
        if self.loss_function_type == "logistic":         # == BCE with logits
            target = torch.full_like(logits_subset, float(target01))
            return F.binary_cross_entropy_with_logits(logits_subset, target, reduction="mean")
        elif self.loss_function_type == "ExponentialLoss":     # exp loss on margin ỹ∈{-1,+1}
            y_tilde = (2.0 * float(target01) - 1.0)   # 0->-1, 1->+1
            return torch.exp(-y_tilde * logits_subset).mean()
        elif self.loss_function_type == "HingeLoss":           # hinge on margin
            y_tilde = (2.0 * float(target01) - 1.0)
            return torch.clamp(1.0 - y_tilde * logits_subset, min=0.0).mean()
        else:
            raise ValueError(f"Unknown loss_function_type: {self.loss_function_type}")

    def compute_Rs(self, X, labels):
        logits = self._logits(X)
        T0_mask, T1_mask, S0_mask = self._split_masks(labels)
        nT0, nT1, nS0 = int(T0_mask.sum()), int(T1_mask.sum()), int(S0_mask.sum())
        if nT0 == 0 or nT1 == 0 or nS0 == 0:
            raise ValueError(f"Empty required group(s): nT0={nT0}, nT1={nT1}, nS0={nS0}.")

        R0T = self._loss_on_subset(logits[T0_mask], target01=0.0)  # target normals
        R1T = self._loss_on_subset(logits[T1_mask], target01=1.0)  # target abnormals
        R0S = self._loss_on_subset(logits[S0_mask], target01=0.0)  # source normals
        return R0T, R1T, R0S

    def f_values(self, R0T, R1T, R0S):
        f1 = R0T - (self.alpha + self.epsilon_0_T)
        f2 = R0S - (self.alpha_prime + self.epsilon_0_S)
        f3 = R1T - (self.R1T_hat_const + self.epsilon_1_T)
        return f1, f2, f3

    def g_value(self, f1, f2, f3):
        f_stack = torch.stack([f1, f2, f3])
        g, idx = torch.max(f_stack, dim=0)
        return g, idx

    def forward(self, X, labels):
        R0T, R1T, R0S = self.compute_Rs(X, labels)
        f1, f2, f3 = self.f_values(R0T, R1T, R0S)
        g, idx = self.g_value(f1, f2, f3)
        return g, idx, (f1, f2, f3), (R0T, R1T, R0S)

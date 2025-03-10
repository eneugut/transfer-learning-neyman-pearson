import torch
import torch.nn as nn

class QuadraticFormModel(nn.Module):
    def __init__(self, input_dim):
        super(QuadraticFormModel, self).__init__()
        self.A = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.v = nn.Parameter(torch.zeros(input_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.A), x.t()).diag() + torch.matmul(self.v, x.t()) + self.b

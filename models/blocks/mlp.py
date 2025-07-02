import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x
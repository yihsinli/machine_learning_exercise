import torch.nn as nn

class MLP_relu(nn.Module):
    def __init__(self, input_dim):
        super(MLP_relu, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class MLP_leakyrelu(nn.Module):
    def __init__(self, input_dim):
        super(MLP_leakyrelu, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)  # 2-class output
    
    def forward(self, x):
        return self.linear(x)  # raw logits

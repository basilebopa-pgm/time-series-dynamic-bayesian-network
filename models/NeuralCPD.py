
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralGaussianCPD(nn.Module):
    def __init__(
        self,
        variable,
        evidence=None,
        hidden_dim=32,
    ):
        super().__init__()

        self.variable = variable
        self.evidence = list(evidence or [])
        self.num_parents = len(self.evidence)

        self.network = nn.Sequential(
            nn.Linear(max(self.num_parents, 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.variance_head = nn.Linear(hidden_dim, 1)

    def forward(self, parents):
        if parents.ndim == 1:
            parents = parents.unsqueeze(0)

        if self.num_parents == 0:
            parents = parents.new_zeros((parents.shape[0], 1))

        hidden = self.network(parents)

        mean = self.mean_head(hidden).squeeze(-1)

        variance = (
            F.softplus(
                self.variance_head(hidden).squeeze(-1)
            ) + 1e-6
        )

        return mean, variance

    def log_prob(self, target, parents):
        mean, variance = self(parents)

        return torch.distributions.Normal(
            mean, torch.sqrt(variance)
        ).log_prob(target)

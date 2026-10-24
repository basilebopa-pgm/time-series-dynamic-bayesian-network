import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralGaussianCPD(nn.Module):
    def __init__(self, variable, evidence=None, hidden_dim=32):
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

        # Normalisation statistics (set from the training data with
        # set_normalization). The network works in standardised space, while
        # forward() and log_prob() stay in the original units.
        self.register_buffer("x_mean", torch.zeros(max(self.num_parents, 1)))
        self.register_buffer("x_std", torch.ones(max(self.num_parents, 1)))
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))

    @torch.no_grad()
    def set_normalization(self, parents, target):
        """Store mean/std of the training inputs and target."""
        if self.num_parents > 0:
            self.x_mean.copy_(parents.mean(dim=0))
            self.x_std.copy_(parents.std(dim=0).clamp_min(1e-6))
        self.y_mean.copy_(target.mean())
        self.y_std.copy_(target.std().clamp_min(1e-6))

    def forward(self, parents):
        if parents.ndim == 1:
            parents = parents.unsqueeze(0)

        if self.num_parents == 0:
            parents = parents.new_zeros((parents.shape[0], 1))
        else:
            parents = (parents - self.x_mean) / self.x_std

        hidden = self.network(parents)

        mean_z = self.mean_head(hidden).squeeze(-1)
        var_z = F.softplus(self.variance_head(hidden).squeeze(-1)) + 1e-6

        # Back to original units
        mean = self.y_mean + self.y_std * mean_z
        variance = var_z * self.y_std ** 2

        return mean, variance

    def log_prob(self, target, parents):
        mean, variance = self(parents)

        return torch.distributions.Normal(mean, torch.sqrt(variance)).log_prob(target)
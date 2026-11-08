
import torch
import torch.nn as nn


class HousingMLP(nn.Module):
    """
    Multilayer Perceptron for Ames Housing price prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.05,
    ):
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Use an initialization matched to ReLU hidden activations."""
        linear_layers = [
            layer for layer in self.network if isinstance(layer, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        # A zero-centred output is appropriate for the standardized target.
        nn.init.xavier_uniform_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


if __name__ == "__main__":
    model = HousingMLP(input_dim=100)

    sample = torch.randn(16, 100)
    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad)
    )

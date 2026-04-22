import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """Soft attention gate for feature selection."""
    def __init__(self, in_channels: int, gate_channels: int):
        super().__init__()
        self.W_x = nn.Conv2d(in_channels, gate_channels, 1, bias=False)
        self.W_g = nn.Conv2d(gate_channels, gate_channels, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(gate_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(torch.relu(self.W_x(x) + self.W_g(g)))
        return x * alpha


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()

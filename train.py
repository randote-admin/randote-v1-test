import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model, loader, optimizer, scheduler, device: str = "cuda") -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, y = batch["input"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)


def build_scheduler(optimizer, epochs: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

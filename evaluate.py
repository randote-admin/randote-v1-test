import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict

@torch.no_grad()
def evaluate(model, loader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        x, y = batch["input"].to(device), batch["label"].to(device)
        out = model(x).argmax(dim=-1)
        preds.extend(out.cpu().numpy())
        labels.extend(y.cpu().numpy())
    preds  = np.array(preds)
    labels = np.array(labels)
    return {
        "f1":        float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
    }

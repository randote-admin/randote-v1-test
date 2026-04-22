"""
LLM 파이프라인 교차분석 테스트 - 20260422_110619
"""
import numpy as np
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    """멀티헤드 어텐션 기반 인코더 블록"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


def train_epoch(model, loader, optimizer, device):
    """단일 에폭 학습 루프"""
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = nn.CrossEntropyLoss()(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

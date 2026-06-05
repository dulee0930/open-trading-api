# -*- coding: utf-8 -*-
"""
tiny_GPT_trading_signal.py

Samsung Electronics historical daily data -> Tiny GPT based trading signal generator.

이 스크립트는 첨부된 `notebook_06.py`의 Tiny GPT 구조를 최대한 활용하되,
원래의 문자 단위 next-token language modeling 문제를 **일별 시장 상태 토큰 시퀀스에서
다음 기간의 BUY/HOLD/SELL 신호를 예측하는 분류 문제**로 바꾼 버전입니다.

주요 설계 의도
-------------
1. `notebook_06.py`의 Head, MultiHeadAttention, FeedForward, Block 구조를 거의 그대로 사용합니다.
2. OHLCV 일봉 데이터에서 추세, 모멘텀, 거래량, RSI, 변동성 상태를 조합해 이산적인
   `market_state_token`을 만듭니다.
3. 실제 트레이더에서는 먼저 `export_history.py`가 기존 CSV에 최신 일봉 데이터를 반영합니다.
   이 스크립트는 그렇게 갱신된 CSV를 입력으로 받아 항상 최신 행 기준 신호를 생성합니다.
4. Tiny GPT는 최근 `block_size`일의 market_state_token을 causal self-attention으로 읽고,
   마지막 hidden state로부터 향후 `horizon`거래일 수익률 기반 신호를 분류합니다.
5. 최신 데이터 구간에 대한 신호를 JSON/CSV로 저장하여 자동매매 모듈이 읽을 수 있게 합니다.

주의: 이 파일은 교육 및 모의투자용 예시입니다. 실제 투자 의사결정에는 별도의 검증,
리스크 관리, 슬리피지/수수료/세금 반영, 과최적화 점검이 필요합니다.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SIGNAL_TO_ID = {"SELL": 0, "HOLD": 1, "BUY": 2}
ID_TO_SIGNAL = {v: k for k, v in SIGNAL_TO_ID.items()}


@dataclass
class ModelConfig:
    """Tiny GPT 모델의 구조 설정입니다.

    기존 검증용 기본값(emb_dim=64, num_layers=2)은 너무 빠른 smoke test에는 좋지만,
    실제 CSV 패턴을 학습하기에는 용량이 부족할 수 있습니다. 기본값을 조금 키우되,
    notebook_06.py의 Tiny GPT 철학을 유지하는 작은 모델 범위 안에 둡니다.
    """

    block_size: int = 64
    emb_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.15


@dataclass
class TrainingConfig:
    """학습 및 라벨 생성 관련 설정입니다.

    `epochs=2` 같은 값은 실행 확인용으로는 충분하지만 예측 확률이 1/3 근처에 머무는
    underfitting을 만들기 쉽습니다. 따라서 기본 epoch를 늘리고 early stopping으로
    과도한 학습 시간을 방지합니다.
    """

    horizon: int = 5
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    val_ratio: float = 0.2
    batch_size: int = 128
    epochs: int = 80
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    min_confidence: float = 0.45
    early_stop_patience: int = 12
    min_epochs: int = 12
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 42


class TradingSignalDataset(Dataset):
    """최근 block_size일의 시장 상태 토큰으로 미래 수익률 기반 신호를 예측하는 Dataset."""

    def __init__(self, token_ids: np.ndarray, labels: np.ndarray, sample_indices: Iterable[int], block_size: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sample_indices = list(sample_indices)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.sample_indices[idx]
        start_idx = end_idx - self.block_size + 1
        x = self.token_ids[start_idx : end_idx + 1]
        y = self.labels[end_idx]
        return x, y


# -----------------------------------------------------------------------------
# Tiny GPT components adapted from notebook_06.py
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """Single causal self-attention head.

    notebook_06.py의 `Head`와 같은 역할을 합니다. 차이는 입력 토큰이 문자가 아니라
    시장 상태 토큰이라는 점뿐입니다.
    """

    def __init__(self, emb_dim: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple causal attention heads followed by projection."""

    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim({emb_dim})은 num_heads({num_heads})로 나누어떨어져야 합니다.")
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feedforward network used inside each Transformer block."""

    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm attention + residual, pre-norm FFN + residual."""

    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPTTradingSignal(nn.Module):
    """Tiny GPT backbone + BUY/HOLD/SELL classification head.

    notebook_06.py의 `TinyGPT`는 각 위치마다 vocabulary logits를 반환했지만,
    여기서는 마지막 위치의 hidden state 하나만 꺼내 `num_classes=3` 신호 logits를 반환합니다.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.signal_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"입력 길이 T={T}가 block_size={self.block_size}보다 큽니다.")
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        last_hidden = h[:, -1, :]
        logits = self.signal_head(last_hidden)
        return logits


# -----------------------------------------------------------------------------
# Feature engineering and tokenization
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder 방식에 가까운 지수평활 RSI를 계산합니다."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def signal_from_future_return(future_return: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:
    """향후 수익률을 BUY/HOLD/SELL 라벨로 변환합니다."""

    return pd.Series(
        np.select(
            [future_return >= buy_threshold, future_return <= sell_threshold],
            ["BUY", "SELL"],
            default="HOLD",
        ),
        index=future_return.index,
    )


def load_and_build_features(csv_path: Path, cfg: TrainingConfig) -> pd.DataFrame:
    """CSV를 읽고 Tiny GPT 입력용 feature frame을 만듭니다."""

    required = ["stck_bsop_date", "stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"]
    df = pd.read_csv(csv_path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["stck_bsop_date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 가격/거래량 별칭. 원본 컬럼명을 유지하면서 모델 코드 가독성을 높입니다.
    df["open"] = df["stck_oprc"].astype(float)
    df["high"] = df["stck_hgpr"].astype(float)
    df["low"] = df["stck_lwpr"].astype(float)
    df["close"] = df["stck_clpr"].astype(float)
    df["volume"] = df["acml_vol"].astype(float)

    # 기본 수익률, 추세, 변동성, 거래량 특징.
    df["ret_1"] = df["close"].pct_change()
    df["logret_1"] = np.log(df["close"]).diff()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["close_ma5_gap"] = df["close"] / df["ma5"] - 1
    df["close_ma20_gap"] = df["close"] / df["ma20"] - 1
    df["close_ma60_gap"] = df["close"] / df["ma60"] - 1
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["atr_proxy"] = df["range_pct"].rolling(14).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    df["rsi14"] = compute_rsi(df["close"], window=14)

    # 현재 일자 기준 horizon 거래일 뒤 종가 수익률을 라벨로 사용합니다.
    df["future_return"] = df["close"].shift(-cfg.horizon) / df["close"] - 1
    df["target_signal"] = signal_from_future_return(df["future_return"], cfg.buy_threshold, cfg.sell_threshold)

    feature_cols = [
        "ret_1",
        "ret_5",
        "ret_20",
        "close_ma5_gap",
        "close_ma20_gap",
        "close_ma60_gap",
        "range_pct",
        "atr_proxy",
        "volume_ratio",
        "rsi14",
        "future_return",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df["market_state_token"] = make_market_state_tokens(df)
    df["label_id"] = df["target_signal"].map(SIGNAL_TO_ID).astype(int)
    return df


def _quinary_bucket(value: float, t1: float, t2: float, t3: float, t4: float, names: List[str]) -> str:
    """연속형 값을 5개 상태로 나누어 최종 선택 노트북의 token 해상도를 보존합니다."""

    if value <= t1:
        return names[0]
    if value <= t2:
        return names[1]
    if value <= t3:
        return names[2]
    if value <= t4:
        return names[3]
    return names[4]


def make_market_state_tokens(df: pd.DataFrame) -> pd.Series:
    """연속형 일봉 특징을 GPT 입력용 5단계 이산 상태 토큰으로 변환합니다.

    사용자가 최종 선택한 `tiny_gpt_trading_signal_real(1).py`의 핵심 변경점인
    5단계 market-state tokenization을 운영용 CLI에서도 그대로 사용합니다.
    """

    trend_labels = ["S_DOWN", "DOWN", "FLAT", "UP", "S_UP"]
    mom_labels = ["S_NEG", "NEG", "NEU", "POS", "S_POS"]
    rsi_labels = ["OVERSOLD", "WEAK", "NEUTRAL", "STRONG", "OVERBOUGHT"]
    volume_labels = ["V_LOW", "LOW", "NORMAL", "HIGH", "V_HIGH"]
    range_labels = ["V_LOW", "LOW", "NORMAL", "HIGH", "V_HIGH"]

    tokens: List[str] = []
    for row in df.itertuples(index=False):
        trend = _quinary_bucket(row.close_ma20_gap, -0.05, -0.015, 0.015, 0.05, trend_labels)
        mom5 = _quinary_bucket(row.ret_5, -0.03, -0.01, 0.01, 0.03, mom_labels)
        mom20 = _quinary_bucket(row.ret_20, -0.06, -0.02, 0.02, 0.06, mom_labels)
        rsi = _quinary_bucket(row.rsi14, 30.0, 45.0, 55.0, 70.0, rsi_labels)
        volume = _quinary_bucket(row.volume_ratio, 0.6, 0.8, 1.2, 1.5, volume_labels)
        volatility = _quinary_bucket(row.atr_proxy, 0.015, 0.025, 0.035, 0.045, range_labels)
        tokens.append(f"T_{trend}|M5_{mom5}|M20_{mom20}|RSI_{rsi}|VOL_{volume}|RNG_{volatility}")
    return pd.Series(tokens, index=df.index)


def build_vocab(tokens: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """market_state_token vocabulary를 만듭니다."""

    vocab = {"<UNK>": 0}
    for token in sorted(set(tokens)):
        vocab[token] = len(vocab)
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab


def encode_tokens(tokens: Iterable[str], vocab: Dict[str, int]) -> np.ndarray:
    return np.array([vocab.get(token, vocab["<UNK>"]) for token in tokens], dtype=np.int64)


# -----------------------------------------------------------------------------
# Training and inference
# -----------------------------------------------------------------------------
def classification_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=class_weights)


def make_class_weights(labels: np.ndarray, device: str) -> torch.Tensor:
    """클래스 불균형 완화를 위한 inverse-frequency class weight."""

    counts = np.bincount(labels, minlength=len(SIGNAL_TO_ID)).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(SIGNAL_TO_ID) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    class_weights: torch.Tensor | None = None,
    max_grad_norm: float = 1.0,
) -> float:
    """notebook_06.py의 train_one_epoch 형태를 분류 문제에 맞게 조정한 함수.

    더 긴 학습에서 gradient 폭주를 막기 위해 gradient clipping을 추가했습니다.
    """

    model.train()
    total_loss, total_count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = classification_cross_entropy(logits, yb, class_weights=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str, class_weights: torch.Tensor | None = None) -> Dict[str, float]:
    model.eval()
    total_loss, total_count, total_correct = 0.0, 0, 0
    total_confidence, total_entropy = 0.0, 0.0
    confusion = np.zeros((len(SIGNAL_TO_ID), len(SIGNAL_TO_ID)), dtype=int)
    pred_counts = np.zeros(len(SIGNAL_TO_ID), dtype=int)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = classification_cross_entropy(logits, yb, class_weights=class_weights)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / math.log(len(SIGNAL_TO_ID))
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        total_correct += (pred == yb).sum().item()
        total_confidence += confidence.sum().item()
        total_entropy += entropy.sum().item()
        for true_id, pred_id in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            confusion[true_id, pred_id] += 1
            pred_counts[pred_id] += 1
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "avg_confidence": total_confidence / max(total_count, 1),
        "avg_normalized_entropy": total_entropy / max(total_count, 1),
    }
    recalls = []
    for signal, sid in SIGNAL_TO_ID.items():
        denom = confusion[sid].sum()
        recall = float(confusion[sid, sid] / denom) if denom else math.nan
        metrics[f"recall_{signal.lower()}"] = recall
        if not math.isnan(recall):
            recalls.append(recall)
        metrics[f"pred_ratio_{signal.lower()}"] = float(pred_counts[sid] / max(total_count, 1))
    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else math.nan
    return metrics


@torch.no_grad()
def predict_one_context(model: nn.Module, context_ids: np.ndarray, device: str, min_confidence: float) -> Dict[str, object]:
    model.eval()
    x = torch.tensor(context_ids[None, :], dtype=torch.long, device=device)
    logits = model(x)
    probs_tensor = F.softmax(logits, dim=-1).squeeze(0)
    probs = probs_tensor.cpu().numpy()
    raw_id = int(np.argmax(probs))
    raw_signal = ID_TO_SIGNAL[raw_id]
    confidence = float(probs[raw_id])
    normalized_entropy = float((-(probs_tensor * torch.log(probs_tensor.clamp_min(1e-12))).sum() / math.log(len(SIGNAL_TO_ID))).cpu().item())

    # confidence가 낮으면 실제 주문 행동은 HOLD로 보수화합니다.
    # raw_signal이 이미 HOLD인 경우에도 confidence_guard_applied=True로 기록해
    # '확신이 낮아 방어 로직이 활성화된 상태'를 명확히 드러냅니다.
    low_confidence = confidence < min_confidence
    guarded_signal = raw_signal if not low_confidence else "HOLD"
    return {
        "raw_signal": raw_signal,
        "trading_signal": guarded_signal,
        "confidence": confidence,
        "prob_sell": float(probs[SIGNAL_TO_ID["SELL"]]),
        "prob_hold": float(probs[SIGNAL_TO_ID["HOLD"]]),
        "prob_buy": float(probs[SIGNAL_TO_ID["BUY"]]),
        "normalized_entropy": normalized_entropy,
        "min_confidence": min_confidence,
        "confidence_guard_applied": bool(low_confidence),
        "action_blocked_by_confidence": bool(low_confidence and raw_signal != "HOLD"),
    }


@torch.no_grad()
def predict_history(model: nn.Module, token_ids: np.ndarray, block_size: int, device: str, batch_size: int = 256) -> pd.DataFrame:
    """전체 학습 가능 구간에 대해 rolling prediction을 생성합니다."""

    model.eval()
    contexts, end_indices = [], []
    for end_idx in range(block_size - 1, len(token_ids)):
        contexts.append(token_ids[end_idx - block_size + 1 : end_idx + 1])
        end_indices.append(end_idx)
    if not contexts:
        return pd.DataFrame(columns=["row_index", "pred_signal", "pred_confidence", "prob_sell", "prob_hold", "prob_buy"])

    probs_list = []
    for start in range(0, len(contexts), batch_size):
        batch = torch.tensor(np.stack(contexts[start : start + batch_size]), dtype=torch.long, device=device)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs_list.append(probs)
    probs_all = np.concatenate(probs_list, axis=0)
    pred_ids = probs_all.argmax(axis=1)
    return pd.DataFrame(
        {
            "row_index": end_indices,
            "pred_signal": [ID_TO_SIGNAL[int(i)] for i in pred_ids],
            "pred_confidence": probs_all.max(axis=1),
            "prob_sell": probs_all[:, SIGNAL_TO_ID["SELL"]],
            "prob_hold": probs_all[:, SIGNAL_TO_ID["HOLD"]],
            "prob_buy": probs_all[:, SIGNAL_TO_ID["BUY"]],
        }
    )


def make_loaders(token_ids: np.ndarray, labels: np.ndarray, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    sample_indices = np.arange(model_cfg.block_size - 1, len(token_ids))
    if len(sample_indices) < 20:
        raise ValueError("학습 샘플이 너무 적습니다. block_size를 줄이거나 더 긴 데이터가 필요합니다.")
    split = int(len(sample_indices) * (1 - train_cfg.val_ratio))
    split = min(max(split, 1), len(sample_indices) - 1)
    train_indices = sample_indices[:split]
    val_indices = sample_indices[split:]

    train_ds = TradingSignalDataset(token_ids, labels, train_indices, model_cfg.block_size)
    val_ds = TradingSignalDataset(token_ids, labels, val_indices, model_cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    return train_loader, val_loader, train_indices, val_indices


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    csv_path = Path(args.csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_history = Path(args.output_history).expanduser().resolve()
    model_output = Path(args.model_output).expanduser().resolve() if args.model_output else None

    model_cfg = ModelConfig(
        block_size=args.block_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_cfg = TrainingConfig(
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_confidence=args.min_confidence,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        min_delta=args.min_delta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )

    set_seed(train_cfg.seed)
    df = load_and_build_features(csv_path, train_cfg)
    vocab, inv_vocab = build_vocab(df["market_state_token"])
    token_ids = encode_tokens(df["market_state_token"], vocab)
    labels = df["label_id"].to_numpy(dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    train_loader, val_loader, train_indices, val_indices = make_loaders(token_ids, labels, model_cfg, train_cfg)

    model = TinyGPTTradingSignal(
        vocab_size=len(vocab),
        block_size=model_cfg.block_size,
        emb_dim=model_cfg.emb_dim,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_cfg.epochs, 1), eta_min=train_cfg.learning_rate * 0.05)
    class_weights = make_class_weights(labels[train_indices], device=device) if args.class_weights else None

    history_log: List[Dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(train_cfg.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=class_weights,
            max_grad_norm=train_cfg.max_grad_norm,
        )
        val_metrics = evaluate_model(model, val_loader, device, class_weights=class_weights)
        scheduler.step()

        # 클래스 불균형이 있으므로 단순 accuracy보다 balanced accuracy를 best model 기준으로 사용합니다.
        current_score = val_metrics.get("balanced_accuracy", val_metrics["accuracy"])
        improved = current_score > best_score + train_cfg.min_delta
        if improved:
            best_score = current_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "is_best": bool(improved),
        }
        history_log.append(row)
        print(
            f"epoch {epoch + 1:03d} | train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f} | "
            f"val bal-acc {val_metrics['balanced_accuracy']:.4f} | "
            f"val conf {val_metrics['avg_confidence']:.4f} | "
            f"val entropy {val_metrics['avg_normalized_entropy']:.4f}"
        )

        if (epoch + 1) >= train_cfg.min_epochs and epochs_without_improvement >= train_cfg.early_stop_patience:
            stopped_early = True
            print(f"early stopping at epoch {epoch + 1}; best epoch={best_epoch}, best balanced accuracy={best_score:.4f}")
            break

    model.load_state_dict(best_state)

    latest_context = token_ids[-model_cfg.block_size :]
    latest_pred = predict_one_context(model, latest_context, device, min_confidence=train_cfg.min_confidence)
    latest_row = df.iloc[-1]

    pred_history = predict_history(model, token_ids, model_cfg.block_size, device)
    history_df = df.reset_index(names="row_index").merge(pred_history, on="row_index", how="left")
    output_history.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_history, index=False, encoding="utf-8-sig")

    result = {
        "symbol": args.symbol,
        "source_csv": str(csv_path),
        "as_of_date": str(latest_row["date"].date()),
        "latest_close": float(latest_row["close"]),
        "latest_market_state_token": str(latest_row["market_state_token"]),
        "data_profile": {
            "row_count_after_feature_drop": int(len(df)),
            "first_feature_date": str(df["date"].iloc[0].date()),
            "last_feature_date": str(df["date"].iloc[-1].date()),
            "required_csv_columns": ["stck_bsop_date", "stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"],
        },
        "target_definition": {
            "horizon_trading_days": train_cfg.horizon,
            "buy_if_future_return_gte": train_cfg.buy_threshold,
            "sell_if_future_return_lte": train_cfg.sell_threshold,
        },
        "prediction": latest_pred,
        "model_config": asdict(model_cfg),
        "training_config": asdict(train_cfg),
        "training_summary": {
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best_score,
            "stopped_early": stopped_early,
            "epochs_ran": len(history_log),
            "class_weights_used": bool(args.class_weights),
        },
        "vocab_size": len(vocab),
        "class_distribution": df["target_signal"].value_counts().to_dict(),
        "validation_metrics_last_epoch": history_log[-1] if history_log else {},
        "validation_metrics_best_epoch": history_log[best_epoch - 1] if best_epoch else {},
        "output_history_csv": str(output_history),
        "input_contract_version": "ai_signal_trader_v1",
        "note": "교육 및 모의투자용 신호입니다. 실제 매매 전 별도 검증과 리스크 관리가 필요합니다.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "training_config": asdict(train_cfg),
                "vocab": vocab,
                "inv_vocab": inv_vocab,
                "signal_to_id": SIGNAL_TO_ID,
            },
            model_output,
        )
        result["model_output"] = str(model_output)

    print("\nLatest trading signal")
    print(json.dumps(result["prediction"], ensure_ascii=False, indent=2))
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved history CSV: {output_history}")
    if model_output is not None:
        print(f"Saved model checkpoint: {model_output}")
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate Samsung trading signal from the CSV refreshed by export_history.py with the selected 5-level Tiny GPT market-state tokenizer.")
    parser.add_argument("--csv", default=str(here / "Samsung_Daily_Data_yfinance.csv"), help="export_history.py로 최신 데이터가 반영된 입력 일봉 CSV 경로")
    parser.add_argument("--symbol", default="005930", help="종목 코드 또는 식별자")
    parser.add_argument("--output-json", default=str(here / "latest_trading_signal.json"), help="최신 신호 JSON 저장 경로")
    parser.add_argument("--output-history", default=str(here / "trading_signals_history.csv"), help="전체 rolling 예측 CSV 저장 경로")
    parser.add_argument("--model-output", default=str(here / "tiny_gpt_trading_signal.pt"), help="모델 체크포인트 저장 경로. 빈 문자열이면 저장하지 않음")

    parser.add_argument("--horizon", type=int, default=5, help="라벨 산출용 미래 거래일 수")
    parser.add_argument("--buy-threshold", type=float, default=0.02, help="미래 수익률이 이 값 이상이면 BUY")
    parser.add_argument("--sell-threshold", type=float, default=-0.02, help="미래 수익률이 이 값 이하이면 SELL")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="이 확률 미만의 raw 신호는 HOLD로 보수화")

    parser.add_argument("--block-size", type=int, default=64, help="Tiny GPT가 보는 최근 거래일 수")
    parser.add_argument("--emb-dim", type=int, default=96, help="토큰 임베딩 차원")
    parser.add_argument("--num-heads", type=int, default=4, help="multi-head attention head 수")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer block 수")
    parser.add_argument("--dropout", type=float, default=0.15, help="dropout 비율")

    parser.add_argument("--epochs", type=int, default=80, help="학습 epoch 수. 실행 확인만 할 때는 2~3으로 낮출 수 있음")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="시간순 validation 비율")
    parser.add_argument("--early-stop-patience", type=int, default=12, help="validation balanced accuracy 개선이 없을 때 조기 종료 patience")
    parser.add_argument("--min-epochs", type=int, default=12, help="조기 종료 전 최소 학습 epoch")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="best score 개선으로 인정할 최소 폭")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm. 0 이하면 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="난수 seed")
    parser.add_argument("--class-weights", dest="class_weights", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false", help="클래스 불균형 보정 loss weight를 사용하지 않음")
    parser.set_defaults(class_weights=True)
    parser.add_argument("--cpu", action="store_true", help="CUDA가 있어도 CPU 사용")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

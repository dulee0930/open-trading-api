"""

tiny_GPT_trading_signal_cli.py



(2).py 기반의 Tiny GPT 트레이딩 신호 생성기 (CLI 버전)

백테스팅/시각화 코드를 제거하고 터미널에서 인자를 받아 실행되도록 구성되었습니다.

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

    block_size: int = 64

    emb_dim: int = 96

    num_heads: int = 4

    num_layers: int = 3

    dropout: float = 0.15



@dataclass

class TrainingConfig:

    horizon: int = 5

    buy_threshold: float = 0.02

    sell_threshold: float = -0.02

    val_ratio: float = 0.2

    batch_size: int = 128

    epochs: int = 20

    learning_rate: float = 5e-4

    weight_decay: float = 1e-2

    min_confidence: float = 0.45

    early_stop_patience: int = 6

    min_epochs: int = 8

    min_delta: float = 1e-4

    max_grad_norm: float = 1.0

    seed: int = 42



class TradingSignalDataset(Dataset):

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



class Head(nn.Module):

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

    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):

        super().__init__()

        if emb_dim % num_heads != 0:

            raise ValueError("emb_dim은 num_heads로 나누어떨어져야 합니다.")

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

    def __init__(self, vocab_size: int, block_size: int, emb_dim: int = 96, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.15, num_classes: int = 3):

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



def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(lower=0.0)

    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50.0)



def signal_from_future_return(future_return: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:

    return pd.Series(

        np.select(

            [future_return >= buy_threshold, future_return <= sell_threshold],

            ["BUY", "SELL"],

            default="HOLD",

        ),

        index=future_return.index,

    )



def _quinary_bucket(value: float, t1: float, t2: float, t3: float, t4: float, names: list) -> str:

    if value <= t1: return names[0]

    elif value <= t2: return names[1]

    elif value <= t3: return names[2]

    elif value <= t4: return names[3]

    else: return names[4]



def make_market_state_tokens(df: pd.DataFrame) -> pd.Series:

    tokens = []

    TREND_LABELS = ["S_DOWN", "DOWN", "FLAT", "UP", "S_UP"]

    MOM_LABELS = ["S_NEG", "NEG", "NEU", "POS", "S_POS"]

    RSI_LABELS = ["OVERSOLD", "WEAK", "NEUTRAL", "STRONG", "OVERBOUGHT"]

    VOL_LABELS = ["V_LOW", "LOW", "NORMAL", "HIGH", "V_HIGH"]

    RNG_LABELS = ["V_LOW", "LOW", "NORMAL", "HIGH", "V_HIGH"]



    for row in df.itertuples(index=False):

        trend = _quinary_bucket(row.close_ma20_gap, -0.05, -0.015, 0.015, 0.05, TREND_LABELS)

        mom5 = _quinary_bucket(row.ret_5, -0.03, -0.01, 0.01, 0.03, MOM_LABELS)

        mom20 = _quinary_bucket(row.ret_20, -0.06, -0.02, 0.02, 0.06, MOM_LABELS)

        rsi = _quinary_bucket(row.rsi14, 30.0, 45.0, 55.0, 70.0, RSI_LABELS)

        volume = _quinary_bucket(row.volume_ratio, 0.6, 0.8, 1.2, 1.5, VOL_LABELS)

        volatility = _quinary_bucket(row.atr_proxy, 0.015, 0.025, 0.035, 0.045, RNG_LABELS)

        token_str = f"T_{trend}|M5_{mom5}|M20_{mom20}|RSI_{rsi}|VOL_{volume}|RNG_{volatility}"

        tokens.append(token_str)

    return pd.Series(tokens, index=df.index)



def load_and_build_features(csv_path: Path, cfg: TrainingConfig) -> pd.DataFrame:

    required = ["stck_bsop_date", "stck_oprc", "stck_hgpr", "stck_lwpr", "stck_clpr", "acml_vol"]

    df = pd.read_csv(csv_path)

    missing = [col for col in required if col not in df.columns]

    if missing:

        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing}")



    df = df.copy()

    df["date"] = pd.to_datetime(df["stck_bsop_date"])

    df = df.sort_values("date").reset_index(drop=True)



    df["open"] = df["stck_oprc"].astype(float)

    df["high"] = df["stck_hgpr"].astype(float)

    df["low"] = df["stck_lwpr"].astype(float)

    df["close"] = df["stck_clpr"].astype(float)

    df["volume"] = df["acml_vol"].astype(float)



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



    df["future_return"] = df["close"].shift(-cfg.horizon) / df["close"] - 1

    df["target_signal"] = signal_from_future_return(df["future_return"], cfg.buy_threshold, cfg.sell_threshold)



    feature_cols = ["ret_1", "ret_5", "ret_20", "close_ma5_gap", "close_ma20_gap", "close_ma60_gap", "range_pct", "atr_proxy", "volume_ratio", "rsi14", "future_return"]

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    df["market_state_token"] = make_market_state_tokens(df)

    df["label_id"] = df["target_signal"].map(SIGNAL_TO_ID).astype(int)

    return df



def build_vocab(tokens: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:

    vocab = {"<UNK>": 0}

    for token in sorted(set(tokens)):

        vocab[token] = len(vocab)

    inv_vocab = {idx: token for token, idx in vocab.items()}

    return vocab, inv_vocab



def encode_tokens(tokens: Iterable[str], vocab: Dict[str, int]) -> np.ndarray:

    return np.array([vocab.get(token, vocab["<UNK>"]) for token in tokens], dtype=np.int64)



def classification_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:

    return F.cross_entropy(logits, targets, weight=class_weights)



def make_class_weights(labels: np.ndarray, device: str) -> torch.Tensor:

    counts = np.bincount(labels, minlength=len(SIGNAL_TO_ID)).astype(float)

    counts = np.maximum(counts, 1.0)

    weights = counts.sum() / (len(SIGNAL_TO_ID) * counts)

    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)



def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, class_weights=None, max_grad_norm: float = 1.0) -> float:

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

def evaluate_model(model: nn.Module, loader: DataLoader, device: str, class_weights=None) -> Dict[str, float]:

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

        if not math.isnan(recall):

            recalls.append(recall)

    metrics["balanced_accuracy"] = float(np.mean(recalls)) if recalls else math.nan

    return metrics



def make_loaders(token_ids: np.ndarray, labels: np.ndarray, model_cfg: ModelConfig, train_cfg: TrainingConfig):

    sample_indices = np.arange(model_cfg.block_size - 1, len(token_ids))

    split = int(len(sample_indices) * (1 - train_cfg.val_ratio))

    split = min(max(split, 1), len(sample_indices) - 1)

    train_indices = sample_indices[:split]

    val_indices = sample_indices[split:]



    train_ds = TradingSignalDataset(token_ids, labels, train_indices, model_cfg.block_size)

    val_ds = TradingSignalDataset(token_ids, labels, val_indices, model_cfg.block_size)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    return train_loader, val_loader, train_indices, val_indices



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

    return pd.DataFrame({

        "row_index": end_indices,

        "pred_signal": [ID_TO_SIGNAL[int(i)] for i in pred_ids],

        "pred_confidence": probs_all.max(axis=1),

        "prob_sell": probs_all[:, SIGNAL_TO_ID["SELL"]],

        "prob_hold": probs_all[:, SIGNAL_TO_ID["HOLD"]],

        "prob_buy": probs_all[:, SIGNAL_TO_ID["BUY"]],

    })



def main():

    parser = argparse.ArgumentParser(description="Generate basic Tiny GPT trading signals (from (2).py logic)")

    parser.add_argument("--csv", required=True, help="입력 일봉 CSV 경로")

    parser.add_argument("--output-json", default="latest_trading_signal.json", help="최신 신호 JSON 저장 경로")

    parser.add_argument("--output-history", default="trading_signals_history.csv", help="전체 롤링 예측 CSV 저장 경로")

    parser.add_argument("--symbol", default="005930", help="종목 코드")

    parser.add_argument("--epochs", type=int, default=20, help="학습 에포크 수")

    args = parser.parse_args()



    csv_path = Path(args.csv)

    output_json = Path(args.output_json)

    output_history = Path(args.output_history)



    model_cfg = ModelConfig()

    train_cfg = TrainingConfig(epochs=args.epochs, min_epochs=8, early_stop_patience=6)



    set_seed(train_cfg.seed)

    df = load_and_build_features(csv_path, train_cfg)

    vocab, inv_vocab = build_vocab(df["market_state_token"])

    token_ids = encode_tokens(df["market_state_token"], vocab)

    labels = df["label_id"].to_numpy(dtype=np.int64)



    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    class_weights = make_class_weights(labels[train_indices], device=device)



    history_log = []

    best_score = -float("inf")

    best_epoch = 0

    best_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0

    stopped_early = False



    print(f"[{args.symbol}] AI 학습을 시작합니다...")

    for epoch in range(train_cfg.epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights=class_weights, max_grad_norm=train_cfg.max_grad_norm)

        val_metrics = evaluate_model(model, val_loader, device, class_weights=class_weights)

        scheduler.step()



        current_score = val_metrics.get("balanced_accuracy", val_metrics["accuracy"])

        improved = current_score > best_score + train_cfg.min_delta

        if improved:

            best_score = current_score

            best_epoch = epoch + 1

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1



        history_log.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,

            "val_balanced_accuracy": current_score

        })



        if (epoch + 1) >= train_cfg.min_epochs and epochs_without_improvement >= train_cfg.early_stop_patience:

            stopped_early = True

            break



    model.load_state_dict(best_state)



    latest_context = token_ids[-model_cfg.block_size :]

    latest_pred = predict_one_context(model, latest_context, device, min_confidence=train_cfg.min_confidence)

    latest_row = df.iloc[-1]



    result = {

        "symbol": args.symbol,

        "source_csv": str(csv_path),

        "as_of_date": str(latest_row["date"].date()),

        "latest_close": float(latest_row["close"]),

        "latest_market_state_token": str(latest_row["market_state_token"]),

        "target_definition": {

            "horizon_trading_days": train_cfg.horizon,

            "buy_if_future_return_gte": train_cfg.buy_threshold,

            "sell_if_future_return_lte": train_cfg.sell_threshold,

        },

        "prediction": latest_pred,

        "training_summary": {

            "best_epoch": best_epoch,

            "best_validation_balanced_accuracy": best_score,

            "stopped_early": stopped_early,

        }

    }



    pred_history = predict_history(model, token_ids, model_cfg.block_size, device)

    history_df = df.reset_index(names="row_index").merge(pred_history, on="row_index", how="left")



    # 파일 저장

    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    

    output_history.parent.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(output_history, index=False, encoding="utf-8-sig")



    print("실행 완료!")

    print(f"최신 신호: {latest_pred['trading_signal']} (확신도: {latest_pred['confidence']:.2f})")

    print(f"저장된 JSON: {output_json}")

    print(f"저장된 CSV: {output_history}")



if __name__ == "__main__":

    main() 


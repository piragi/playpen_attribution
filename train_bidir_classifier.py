from __future__ import annotations

"""SAE fingerprint classifier: predict high vs low attribution from SAE features.

Architecture: mean-pooled set encoder over top-K (feature_id, activation) pairs.
  embed(feat_id) * feat_val  →  Linear+GELU+Dropout  →  masked mean  →  head

Runs two ablation sweeps:
  K ablation: K ∈ {64, 128, 256, 512}   — how many top features are needed
  Baselines A/B for reference

Usage:
    uv run train_bidir_classifier.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

CONFIG = {
    "sae_features_dir": "runs/smoltalk_v1/sae_features/layer12_width16k",
    "sae_id": "layer_12_width_16k_l0_small",
    "d_sae": 16384,
    "output_dir": "runs/smoltalk_v1/sae_classifier",
    # Model hyperparameters
    "d_embed": 64,
    "d_hidden": 128,
    "dropout": 0.1,
    # K ablation values to sweep
    "k_values": [64, 128, 256, 512],
    # Training
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,
    "val_frac": 0.20,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SAEDataset(Dataset):
    """Loads pre-extracted SAE feature arrays from a .npz file."""

    def __init__(self, npz_path: str, k: int) -> None:
        data = np.load(npz_path)
        # Slice to the requested K (features are already top-sorted by activation)
        self.feat_ids = torch.from_numpy(data["feat_ids"][:, :k].astype(np.int64))
        self.feat_vals = torch.from_numpy(data["feat_vals"][:, :k].astype(np.float32))
        self.masks = torch.from_numpy(data["masks"][:, :k].astype(bool))
        self.labels = torch.from_numpy(data["labels"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "feat_ids": self.feat_ids[idx],
            "feat_vals": self.feat_vals[idx],
            "mask": self.masks[idx],
            "label": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAEFingerprint(nn.Module):
    """Mean-pooled set encoder over (feature_id, activation) pairs."""

    def __init__(self, d_sae: int, d_embed: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(d_sae, d_embed)
        self.proj = nn.Sequential(
            nn.Linear(d_embed, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        feat_ids: torch.Tensor,   # (B, K)
        feat_vals: torch.Tensor,  # (B, K)
        mask: torch.Tensor,       # (B, K) bool
    ) -> torch.Tensor:
        x = self.embed(feat_ids) * feat_vals.unsqueeze(-1)    # (B, K, d_embed)
        h = self.proj(x)                                       # (B, K, d_hidden)
        mask_f = mask.float().unsqueeze(-1)                    # (B, K, 1)
        pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, d_hidden)
        return self.head(pooled).squeeze(-1)                   # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_neural(
    model: SAEFingerprint,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    out_dir: Path,
    device: str,
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0
    patience_left = cfg["patience"]
    history: list[dict] = []

    for epoch in range(cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            feat_ids = batch["feat_ids"].to(device)
            feat_vals = batch["feat_vals"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(feat_ids, feat_vals, mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        val_auroc, val_acc = evaluate_neural(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc, "val_acc": val_acc})

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_left = cfg["patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"    early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  loss={train_loss:.4f}  val_auroc={val_auroc:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    val_auroc, val_acc = evaluate_neural(model, val_loader, device)
    return {"val_auroc": val_auroc, "val_acc": val_acc, "history": history}


def evaluate_neural(model: SAEFingerprint, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            feat_ids = batch["feat_ids"].to(device)
            feat_vals = batch["feat_vals"].to(device)
            mask = batch["mask"].to(device)
            logits = model(feat_ids, feat_vals, mask).cpu()
            all_logits.extend(logits.tolist())
            all_labels.extend(batch["label"].tolist())

    labels_arr = np.array(all_labels)
    probs_arr = torch.sigmoid(torch.tensor(all_logits)).numpy()
    auroc = float(roc_auc_score(labels_arr, probs_arr))
    acc = float(((probs_arr >= 0.5).astype(float) == labels_arr).mean())
    return auroc, acc


# ---------------------------------------------------------------------------
# Sklearn baselines (reference)
# ---------------------------------------------------------------------------

def run_baselines(npz_path: Path, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    data = np.load(npz_path)
    all_labels = data["labels"]
    y_train = all_labels[train_idx].astype(float)
    y_val = all_labels[val_idx].astype(float)

    results: dict[str, dict] = {}

    # A: global stats only
    stats = data["global_stats"]
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(stats[train_idx])
    x_va = scaler.transform(stats[val_idx])
    clf_a = LogisticRegression(max_iter=1000, C=1.0)
    clf_a.fit(x_tr, y_train)
    probs_a = clf_a.predict_proba(x_va)[:, 1]
    results["A_logreg_stats"] = {
        "val_auroc": float(roc_auc_score(y_val, probs_a)),
        "val_acc": float(((probs_a >= 0.5) == y_val).mean()),
    }

    # B: L1 on dense 16k sparse vector
    feat_ids = data["feat_ids"]
    feat_vals = data["feat_vals"]
    d_sae = data["feat_ids"].max() + 1  # conservative upper bound
    n = feat_ids.shape[0]
    dense = np.zeros((n, int(d_sae)), dtype=np.float32)
    for i in range(n):
        dense[i, feat_ids[i]] = feat_vals[i]
    clf_b = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    clf_b.fit(dense[train_idx], y_train)
    probs_b = clf_b.predict_proba(dense[val_idx])[:, 1]
    results["B_l1_sparse"] = {
        "val_auroc": float(roc_auc_score(y_val, probs_b)),
        "val_acc": float(((probs_b >= 0.5) == y_val).mean()),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    npz_path = Path(cfg["sae_features_dir"]) / f"{cfg['sae_id']}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"SAE features not found: {npz_path}\nRun sae_analysis.py first.")

    # Use K=512 dataset to determine the train/val split (same indices for all K runs)
    max_k = max(cfg["k_values"])
    full_ds = SAEDataset(str(npz_path), k=max_k)
    n_total = len(full_ds)
    n_val = int(round(n_total * cfg["val_frac"]))
    n_train = n_total - n_val
    print(f"Dataset: {n_total} samples ({n_train} train, {n_val} val)")

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    train_idx = np.array(train_ds.indices)  # type: ignore[attr-defined]
    val_idx = np.array(val_ds.indices)      # type: ignore[attr-defined]

    all_results: dict[str, dict] = {}

    # --- Baselines ---
    print("\n--- Baselines ---")
    baseline_results = run_baselines(npz_path, train_idx, val_idx)
    for name, res in baseline_results.items():
        all_results[name] = res
        print(f"  [{name}]  auroc={res['val_auroc']:.4f}  acc={res['val_acc']:.4f}")

    # --- K ablation ---
    print("\n--- K ablation (mean pooling, no global stats) ---")
    for k in cfg["k_values"]:
        print(f"\n[K={k}]")
        out_dir = out_root / f"K{k}"
        out_dir.mkdir(exist_ok=True)

        ds_k = SAEDataset(str(npz_path), k=k)
        train_ds_k = torch.utils.data.Subset(ds_k, train_idx.tolist())
        val_ds_k = torch.utils.data.Subset(ds_k, val_idx.tolist())
        train_loader = DataLoader(train_ds_k, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds_k, batch_size=cfg["batch_size"], shuffle=False)

        model = SAEFingerprint(
            d_sae=cfg["d_sae"],
            d_embed=cfg["d_embed"],
            d_hidden=cfg["d_hidden"],
            dropout=cfg["dropout"],
        )
        result = train_neural(model, train_loader, val_loader, cfg, out_dir, device)
        all_results[f"K{k}"] = {"val_auroc": result["val_auroc"], "val_acc": result["val_acc"], "K": k}
        print(f"    best  auroc={result['val_auroc']:.4f}  acc={result['val_acc']:.4f}")
        (out_dir / "history.json").write_text(json.dumps(result["history"], indent=2))

    # --- Summary ---
    results_path = out_root / "ablation_results.json"
    results_path.write_text(json.dumps({"config": cfg, "n_train": n_train, "n_val": n_val, "results": all_results}, indent=2))

    print("\n=== Results ===")
    print(f"{'Run':<18} {'AUROC':>7} {'Acc':>7}")
    for name, res in all_results.items():
        print(f"{name:<18} {res['val_auroc']:>7.4f} {res['val_acc']:>7.4f}")
    print(f"\nSaved → {results_path}")


if __name__ == "__main__":
    main()

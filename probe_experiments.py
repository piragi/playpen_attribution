"""Probe experiments: test different pooling strategies and residualization.

Extracts embeddings with multiple pooling methods (last-token, mean-response,
first-response, multi-layer) and runs a battery of Ridge probes to disentangle
length signal from genuine quality signal in Bergson attribution scores.

Results are saved to runs/smoltalk_v5/probe/experiments/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from pipeline_common import (
    ensure_hf_home_env,
    get_transformer_layers_for_hook,
    last_response_token_positions,
    pad_tokenized_batch,
    resolve_device_dtype,
)

ensure_hf_home_env()

OUT_DIR = Path("runs/smoltalk_v5/probe/experiments_individual")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Cache dir for embeddings (shared across experiments)
EMB_CACHE_DIR = Path("runs/smoltalk_v5/probe/experiments")
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = json.loads(Path("runs/smoltalk_v5/manifest.json").read_text())
BASE_MODEL = MANIFEST["base_model"]
POOL_PATH = MANIFEST["splits"]["score_pool"]["path"]
SCORES_PATH = Path("runs/smoltalk_v5/scores_individual/row_diagnostics.jsonl")
BATCH_SIZE = 16
SEED = 42
RIDGE_ALPHA = 1.0
VAL_FRAC = 0.20


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_multi_pool(
    pool_ds,
    base_model: str,
    layers: list[int],
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract embeddings with multiple pooling strategies in one pass per layer.

    Returns dict keyed like "last_L17", "mean_resp_L17", "first_resp_L17".
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ATTN = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, attn_implementation=ATTN
    )
    model.eval()
    model.config.use_cache = False

    results: dict[str, list[np.ndarray]] = {}
    n = len(pool_ds)

    for layer in layers:
        captured: dict[str, torch.Tensor] = {}

        def _hook(_m, _i, out, _captured=captured):
            _captured["acts"] = out[0] if isinstance(out, tuple) else out

        handle = get_transformer_layers_for_hook(model)[layer].register_forward_hook(_hook)

        key_last = f"last_L{layer}"
        key_mean = f"mean_resp_L{layer}"
        key_first = f"first_resp_L{layer}"
        results[key_last] = []
        results[key_mean] = []
        results[key_first] = []

        print(f"  Extracting layer {layer} ...", flush=True)
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                batch = pool_ds.select(range(start, min(start + batch_size, n)))
                ids_t, lbl_t = pad_tokenized_batch(
                    batch["input_ids"], batch["labels"], device=device
                )
                # Only need hidden states up to our hook layer, but we can't
                # easily stop early. Instead just discard the output to save
                # the lm_head allocation.
                out = model(input_ids=ids_t, output_hidden_states=False)
                del out
                hidden = captured["acts"].detach()  # (B, T, D)

                # Response mask: where labels != -100
                resp_mask = lbl_t.ne(-100).to(hidden.device)  # (B, T)

                # 1) Last response token (original strategy)
                last_pos = last_response_token_positions(lbl_t)
                idx = last_pos.to(hidden.device).unsqueeze(-1).unsqueeze(-1)
                idx = idx.expand(-1, 1, hidden.shape[-1])
                last_pooled = hidden.gather(dim=1, index=idx).squeeze(1)
                results[key_last].append(last_pooled.float().cpu().numpy())

                # 2) Mean over response tokens
                mask_f = resp_mask.unsqueeze(-1).float()  # (B, T, 1)
                mean_pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                results[key_mean].append(mean_pooled.float().cpu().numpy())

                # 3) First response token
                # argmax on the mask gives the first True position
                first_pos = resp_mask.to(torch.int64).argmax(dim=1)
                fidx = first_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.shape[-1])
                first_pooled = hidden.gather(dim=1, index=fidx).squeeze(1)
                results[key_first].append(first_pooled.float().cpu().numpy())

                del hidden
                torch.cuda.empty_cache()

                if start > 0 and start % (batch_size * 50) == 0:
                    print(f"    {start + batch_size:,}/{n:,}", flush=True)

        handle.remove()
        torch.cuda.empty_cache()

    # Concatenate
    return {k: np.concatenate(v, axis=0) for k, v in results.items()}


# ── Probing ──────────────────────────────────────────────────────────────────

def run_probe(name: str, X: np.ndarray, y: np.ndarray, train_idx, val_idx, alpha=RIDGE_ALPHA) -> dict:
    Xtr, Xval = X[train_idx], X[val_idx]
    ytr, yval = y[train_idx], y[val_idx]
    probe = Ridge(alpha=alpha)
    probe.fit(Xtr, ytr)
    pred_val = probe.predict(Xval)
    r2 = float(probe.score(Xval, yval))
    r, p = pearsonr(yval, pred_val)
    r2_tr = float(probe.score(Xtr, ytr))
    r_tr, _ = pearsonr(ytr, probe.predict(Xtr))
    result = {
        "name": name,
        "val_R2": round(r2, 4),
        "val_r": round(float(r), 4),
        "val_p": float(p),
        "train_R2": round(r2_tr, 4),
        "train_r": round(float(r_tr), 4),
    }
    print(f"  {name:55s}  val_R2={r2:+.4f}  val_r={float(r):+.4f}  train_R2={r2_tr:.4f}")
    return result


def residualize_against_length(X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    lr = LinearRegression().fit(lengths.reshape(-1, 1), X)
    return X - lr.predict(lengths.reshape(-1, 1))


def residualize_scores(y: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    lr = LinearRegression().fit(lengths.reshape(-1, 1), y)
    return y - lr.predict(lengths.reshape(-1, 1))


def run_all_experiments(
    embeddings: dict[str, np.ndarray],
    scores: np.ndarray,
    lengths: np.ndarray,
) -> list[dict]:
    idx = np.arange(len(scores))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED)

    scores_resid = residualize_scores(scores, lengths)
    print(f"  r(resid_score, length) = {pearsonr(scores_resid, lengths)[0]:.6f} (should be ~0)")

    all_results = []

    # Baseline: length alone
    print("\n--- Length-only baselines ---")
    all_results.append(run_probe("length_only → score", lengths.reshape(-1, 1), scores, train_idx, val_idx, alpha=1.0))
    all_results.append(run_probe("length_only → resid_score", lengths.reshape(-1, 1), scores_resid, train_idx, val_idx, alpha=1.0))

    for emb_name, emb in embeddings.items():
        print(f"\n--- {emb_name} ---")
        emb_resid = residualize_against_length(emb, lengths)

        # Raw embeddings → bergson score
        all_results.append(run_probe(f"{emb_name} → score", emb, scores, train_idx, val_idx))

        # How well do embeddings predict length?
        all_results.append(run_probe(f"{emb_name} → length", emb, lengths, train_idx, val_idx))

        # Residualized embeddings → bergson score
        all_results.append(run_probe(f"{emb_name}_resid → score", emb_resid, scores, train_idx, val_idx))

        # Raw embeddings → residualized score
        all_results.append(run_probe(f"{emb_name} → resid_score", emb, scores_resid, train_idx, val_idx))

        # Residualized embeddings → residualized score
        all_results.append(run_probe(f"{emb_name}_resid → resid_score", emb_resid, scores_resid, train_idx, val_idx))

        # Partial correlation: probe prediction vs score, controlling for length
        probe = Ridge(alpha=RIDGE_ALPHA).fit(emb[train_idx], scores[train_idx])
        pred_val = probe.predict(emb[val_idx])
        len_val = lengths[val_idx]
        score_val = scores[val_idx]
        pred_r = residualize_scores(pred_val, len_val)
        score_r = residualize_scores(score_val, len_val)
        partial_r, partial_p = pearsonr(pred_r, score_r)
        print(f"  {'partial r(pred, score | length)':55s}  = {partial_r:+.4f}  (p={partial_p:.2e})")
        all_results.append({
            "name": f"{emb_name} partial_r(pred,score|length)",
            "val_r": round(float(partial_r), 4),
            "val_p": float(partial_p),
        })

    # Cross-validated check on the most interesting embeddings (mean_resp, early layers)
    # to verify results aren't a fluke of the single split
    from sklearn.model_selection import KFold
    print("\n" + "=" * 90)
    print("CROSS-VALIDATION (5-fold) on resid_score target")
    print("=" * 90)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for emb_name in sorted(embeddings.keys()):
        if not emb_name.startswith("mean_resp"):
            continue
        emb = embeddings[emb_name]
        fold_r2s = []
        fold_rs = []
        for fold_train, fold_val in kf.split(emb):
            probe = Ridge(alpha=RIDGE_ALPHA).fit(emb[fold_train], scores_resid[fold_train])
            r2 = float(probe.score(emb[fold_val], scores_resid[fold_val]))
            r, _ = pearsonr(scores_resid[fold_val], probe.predict(emb[fold_val]))
            fold_r2s.append(r2)
            fold_rs.append(float(r))
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_r = np.mean(fold_rs)
        print(f"  {emb_name:30s}  R2={mean_r2:+.4f} ± {std_r2:.4f}  r={mean_r:+.4f}  folds={[f'{x:+.3f}' for x in fold_r2s]}")
        all_results.append({
            "name": f"CV5 {emb_name} → resid_score",
            "val_R2_mean": round(mean_r2, 4),
            "val_R2_std": round(std_r2, 4),
            "val_r_mean": round(mean_r, 4),
            "fold_R2s": [round(x, 4) for x in fold_r2s],
        })

    # Also CV the last-token embeddings for comparison
    for emb_name in sorted(embeddings.keys()):
        if not emb_name.startswith("last_L"):
            continue
        if "cached" in emb_name:
            continue
        emb = embeddings[emb_name]
        fold_r2s = []
        fold_rs = []
        for fold_train, fold_val in kf.split(emb):
            probe = Ridge(alpha=RIDGE_ALPHA).fit(emb[fold_train], scores_resid[fold_train])
            r2 = float(probe.score(emb[fold_val], scores_resid[fold_val]))
            r, _ = pearsonr(scores_resid[fold_val], probe.predict(emb[fold_val]))
            fold_r2s.append(r2)
            fold_rs.append(float(r))
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_r = np.mean(fold_rs)
        print(f"  {emb_name:30s}  R2={mean_r2:+.4f} ± {std_r2:.4f}  r={mean_r:+.4f}  folds={[f'{x:+.3f}' for x in fold_r2s]}")
        all_results.append({
            "name": f"CV5 {emb_name} → resid_score",
            "val_R2_mean": round(mean_r2, 4),
            "val_R2_std": round(std_r2, 4),
            "val_r_mean": round(mean_r, 4),
            "fold_R2s": [round(x, 4) for x in fold_r2s],
        })

    # Alpha sensitivity check on the best candidate
    print("\n--- Alpha sensitivity (mean_resp_L8 → resid_score) ---")
    if "mean_resp_L8" in embeddings:
        emb = embeddings["mean_resp_L8"]
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            all_results.append(run_probe(
                f"mean_resp_L8 → resid_score (α={alpha})",
                emb, scores_resid, train_idx, val_idx, alpha=alpha
            ))

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device, dtype = resolve_device_dtype()
    print(f"Device: {device}, dtype: {dtype}")

    pool_ds = load_from_disk(POOL_PATH)
    print(f"Pool: {len(pool_ds):,} examples")

    lengths = np.array(pool_ds["length"], dtype=np.float32)
    diag = sorted(
        [json.loads(l) for l in SCORES_PATH.read_text().splitlines() if l.strip()],
        key=lambda r: r["index"],
    )
    scores = np.array([r["score"] for r in diag], dtype=np.float32)

    # Check for cached last-token embeddings
    cached_last = Path("runs/smoltalk_v5/probe/pool_embeddings_71baad84.npy")

    layers = [2, 4, 6, 8, 10, 12, 17, 22]

    # Try to load cached embeddings first; only extract missing ones
    embeddings: dict[str, np.ndarray] = {}
    layers_to_extract = []
    for layer in layers:
        all_cached = True
        for pool_type in ["last", "mean_resp", "first_resp"]:
            key = f"{pool_type}_L{layer}"
            cache_path = EMB_CACHE_DIR / f"emb_{key}.npy"
            if cache_path.exists():
                embeddings[key] = np.load(str(cache_path))
                print(f"  Loaded cached {key}")
            else:
                all_cached = False
        if not all_cached:
            layers_to_extract.append(layer)

    if layers_to_extract:
        print(f"\nExtracting embeddings for layers {layers_to_extract} ...")
        t0 = time.time()
        new_emb = extract_multi_pool(pool_ds, BASE_MODEL, layers_to_extract, device, dtype, BATCH_SIZE)
        print(f"Extraction took {time.time() - t0:.1f}s")
        for name, emb in new_emb.items():
            np.save(str(EMB_CACHE_DIR / f"emb_{name}.npy"), emb)
            embeddings[name] = emb
    else:
        print("\nAll embeddings loaded from cache.")

    # Also load the original cached last-token L17 (from adapter) for comparison
    if cached_last.exists():
        embeddings["last_L17_cached"] = np.load(str(cached_last))

    print(f"Total embedding matrices: {len(embeddings)}")

    # Run experiments
    print("\n" + "=" * 90)
    print("PROBING EXPERIMENTS")
    print("=" * 90)
    results = run_all_experiments(embeddings, scores, lengths)

    # Save results
    results_path = OUT_DIR / "probe_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY: val_R2 for key experiments")
    print("=" * 90)
    key_experiments = [r for r in results if "val_R2" in r]
    key_experiments.sort(key=lambda r: r.get("val_R2", -99), reverse=True)
    for r in key_experiments:
        print(f"  {r['val_R2']:+.4f}  {r['name']}")


if __name__ == "__main__":
    main()

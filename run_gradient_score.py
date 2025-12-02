import torch
from pathlib import Path
from bergson import IndexConfig, DataConfig
from bergson.config import ScoreConfig
from bergson.build import build
from bergson.score.score import score_dataset

MODEL_NAME = "EleutherAI/pythia-14m"
DATASET_NAME = "NeelNanda/pile-10k"
GENERAL_SIZE = 10000  # General preconditioner built on full dataset
SPECIFIC_SIZE = 1000  # Specific preconditioner and self-influence test on subset
GENERAL_INDEX = "general_index"
SPECIFIC_INDEX = "specific_index"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build general preconditioner on full dataset
if not Path(GENERAL_INDEX).exists():
    print(f"Building general preconditioner on {GENERAL_SIZE} examples...")
    general_data_cfg = DataConfig(dataset=DATASET_NAME, split=f"train[:{GENERAL_SIZE}]", prompt_column="text", truncation=True)
    general_cfg = IndexConfig(run_path=GENERAL_INDEX, model=MODEL_NAME, data=general_data_cfg, projection_dim=16, projection_type="rademacher")
    build(general_cfg)

# Build specific preconditioner on subset
if not Path(SPECIFIC_INDEX).exists():
    print(f"Building specific preconditioner on {SPECIFIC_SIZE} examples...")
    specific_data_cfg = DataConfig(dataset=DATASET_NAME, split=f"train[:{SPECIFIC_SIZE}]", prompt_column="text", truncation=True)
    specific_cfg = IndexConfig(run_path=SPECIFIC_INDEX, model=MODEL_NAME, data=specific_data_cfg, projection_dim=16, projection_type="rademacher")
    build(specific_cfg)

# For self-influence, use training data as queries (no separate query index needed)
# Score each training example against all training examples (including itself)
import numpy as np
import json
from datasets import load_dataset

print("\nRunning self-influence evaluation with different mixing coefficients...")
print(f"General preconditioner: {GENERAL_SIZE} examples")
print(f"Specific preconditioner + test: {SPECIFIC_SIZE} examples")
data_cfg = DataConfig(dataset=DATASET_NAME, split=f"train[:{SPECIFIC_SIZE}]", prompt_column="text", truncation=True)

results = []
per_query_ranks = []  # Track each query's rank at each mixing coefficient

for mixing in range(0, 11, 1):
    mixing_val = mixing / 10.
    scores_path = f"scores_output_mix_{mixing}"

    print(f"\n{'='*80}")
    print(f"Testing mixing_coefficient = {mixing_val:.1f}")
    print(f"{'='*80}")

    index_cfg = IndexConfig(run_path=scores_path, model=MODEL_NAME, data=data_cfg, projection_dim=16, projection_type="rademacher")
    score_cfg = ScoreConfig(
        query_path=SPECIFIC_INDEX,  # Use specific subset as queries
        score="individual",
        unit_normalize=True,
        index_preconditioner_path=SPECIFIC_INDEX,  # Task-specific landscape
        query_preconditioner_path=GENERAL_INDEX,    # General landscape (mixed via coefficient)
        mixing_coefficient=mixing_val,
    )
    score_dataset(index_cfg, score_cfg)

    # Load and analyze self-influence scores
    scores_file = Path(scores_path) / "scores.bin"
    info_file = Path(scores_path) / "info.json"

    with open(info_file) as f:
        info = json.load(f)

    scores_mmap = np.memmap(str(scores_file), dtype=np.dtype(info["dtype"]), mode="r", shape=(info["num_items"],))
    scores_matrix = np.array([[scores_mmap[f"score_{j}"][i] for j in range(SPECIFIC_SIZE)] for i in range(SPECIFIC_SIZE)])

    # Calculate self-influence ranks
    self_influence_ranks = []
    for query_idx in range(SPECIFIC_SIZE):
        query_scores = scores_matrix[:, query_idx]
        ranked_indices = np.argsort(query_scores)[::-1]
        self_rank = np.where(ranked_indices == query_idx)[0][0] + 1
        self_influence_ranks.append(self_rank)

    per_query_ranks.append(self_influence_ranks)

    # Calculate metrics
    mean_rank = np.mean(self_influence_ranks)
    median_rank = np.median(self_influence_ranks)
    std_rank = np.std(self_influence_ranks)
    top1_accuracy = np.mean([r == 1 for r in self_influence_ranks])
    top5_accuracy = np.mean([r <= 5 for r in self_influence_ranks])
    top10_accuracy = np.mean([r <= 10 for r in self_influence_ranks])

    results.append({
        'mixing': mixing_val,
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'std_rank': std_rank,
        'top1': top1_accuracy,
        'top5': top5_accuracy,
        'top10': top10_accuracy,
    })

# Print aggregate comparison table
print(f"\n\n{'='*110}")
print("AGGREGATE MIXING COEFFICIENT COMPARISON")
print(f"{'='*110}")
print(f"{'Mixing':<8} {'Mean':<8} {'±Std':<8} {'Median':<8} {'Top-1':<12} {'Top-5':<12} {'Top-10':<12}")
print(f"{'-'*110}")
for r in results:
    print(f"{r['mixing']:<8.1f} {r['mean_rank']:<8.2f} {r['std_rank']:<8.2f} {r['median_rank']:<8.1f} "
          f"{r['top1']*100:<11.1f}% {r['top5']*100:<11.1f}% {r['top10']*100:<11.1f}%")

best = min(results, key=lambda x: x['mean_rank'])
print(f"\n✓ Best overall mixing coefficient: {best['mixing']:.1f} (mean rank: {best['mean_rank']:.2f} ±{best['std_rank']:.2f})")

# Analyze per-query variability
print(f"\n\n{'='*100}")
print("PER-QUERY ANALYSIS: Which mixing coefficient works best for each query?")
print(f"{'='*100}")

per_query_ranks = np.array(per_query_ranks)  # Shape: (num_mixing, num_queries)
best_mixing_per_query = np.argmin(per_query_ranks, axis=0) / 10.  # Best mixing for each query

print(f"\nDistribution of best mixing coefficients across {SPECIFIC_SIZE} queries:")
print(f"(Shows which mixing gave each query its BEST rank, not necessarily rank=1)")
for mixing in range(0, 11):
    mixing_val = mixing / 10.
    mask = best_mixing_per_query == mixing_val
    count = np.sum(mask)
    pct = 100 * count / SPECIFIC_SIZE
    bar = '█' * int(pct / 2)

    # Show average rank of queries that prefer this mixing
    if count > 0:
        avg_rank_at_best = np.mean(per_query_ranks[mixing, mask])
        print(f"  {mixing_val:.1f}: {count:4d} queries ({pct:5.1f}%) {bar} → avg rank {avg_rank_at_best:.1f}")
    else:
        print(f"  {mixing_val:.1f}: {count:4d} queries ({pct:5.1f}%) {bar}")

# Show variance in rank improvement per query
print(f"\nPer-query rank variability across mixing coefficients:")
rank_ranges = np.max(per_query_ranks, axis=0) - np.min(per_query_ranks, axis=0)
mean_range = np.mean(rank_ranges)
median_range = np.median(rank_ranges)
print(f"  Mean rank range per query: {mean_range:.2f}")
print(f"  Median rank range per query: {median_range:.2f}")
print(f"  (Range = best_rank - worst_rank for each query across all mixing coefficients)")

high_variance_queries = np.where(rank_ranges > np.percentile(rank_ranges, 75))[0]
print(f"\n  {len(high_variance_queries)} queries ({100*len(high_variance_queries)/SPECIFIC_SIZE:.1f}%) show high sensitivity to mixing coefficient")

# Show examples of queries with different preferences
dataset = load_dataset(DATASET_NAME, split="train")
train_data = dataset.select(range(SPECIFIC_SIZE))

print(f"\n{'='*100}")
print("SAMPLE QUERIES WITH DIFFERENT MIXING PREFERENCES (first 5)")
print(f"{'='*100}")
for i in range(min(5, SPECIFIC_SIZE)):
    best_mix = best_mixing_per_query[i]
    ranks_for_query = per_query_ranks[:, i]
    text = train_data[int(i)]["text"][:80]
    print(f"\nQuery {i}: Best at mixing={best_mix:.1f}")
    print(f"  Text: {text}...")
    print(f"  Ranks: ", end="")
    for mixing_idx, rank in enumerate(ranks_for_query):
        print(f"{mixing_idx/10:.1f}→#{rank}", end="  ")
    print()




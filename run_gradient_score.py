import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from bergson import IndexConfig, collect_gradients, GradientProcessor, Attributor

MODEL_NAME = "EleutherAI/pythia-14m"
DATASET_NAME = "NeelNanda/pile-10k"
NUM_EXAMPLES = 1000  # Use more for better results
MAX_LENGTH = 512  # Keep consistent for build and query
INDEX_PATH = "gradient_index"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_dataset(DATASET_NAME, split="train")
data_subset = dataset.select(range(NUM_EXAMPLES))

tokenize_function = lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )
tokenized = data_subset.map(tokenize_function, batched=True, remove_columns=data_subset.column_names, desc="Tokenizing")
tokenized = tokenized.map( lambda x: {"labels": x["input_ids"].copy()}, desc="Adding labels")

processor = GradientProcessor(projection_dim=16, projection_type="rademacher")
cfg = IndexConfig(run_path=INDEX_PATH, skip_preconditioners=False)
collect_gradients(model, tokenized, processor, cfg)

query_subset = dataset.select(range(NUM_EXAMPLES, NUM_EXAMPLES + 32)




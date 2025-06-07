import os
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch

# Configuration
root_path = "clone_detection_checkpoints_no_tokenize/fold_0/train/transform"
batch_size = 32
shuffle = True

# 1. Load local dataset and extract question_pair_ids
chunk_dirs = [
    d for d in os.listdir(root_path) 
    if d.startswith("chunk_") and os.path.isdir(os.path.join(root_path, d))
]
arrow_files = [
    os.path.join(root_path, chunk_dir, "data-00000-of-00001.arrow")
    for chunk_dir in chunk_dirs
    if os.path.exists(os.path.join(root_path, chunk_dir, "data-00000-of-00001.arrow"))
]

if not arrow_files:
    raise FileNotFoundError("No arrow files found in chunk directories")

local_dataset = load_dataset("arrow", data_files=arrow_files, split="train")

if "question_pair_id" not in local_dataset.column_names:
    raise ValueError("Local dataset missing 'question_pair_id' field")

# Extract unique question_pair_ids and convert to float (matching HF dataset format)
local_question_pair_ids = set()
local_pair_ids = set()
for example in local_dataset:
    if example["question_pair_id"] is not None:
        local_question_pair_ids.add(example["question_pair_id"])
    if example["pair_id"] is not None:
        local_pair_ids.add(example["pair_id"])
print(f"Found {len(local_question_pair_ids)} unique question_pair_ids in local dataset")
print(f"Found {len(local_pair_ids)} unique local_pair_ids in local dataset")

# 2. Load and filter Hugging Face dataset
try:
    hf_dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold", split="train")
except ValueError:
    # Handle dataset variants if default split doesn't work
    hf_dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold")["train"]

# Verify required columns exist in HF dataset
required_columns = {"code1", "code2", "similar", "question_pair_id"}
missing_columns = required_columns - set(hf_dataset.column_names)
if missing_columns:
    raise ValueError(f"HF dataset missing required columns: {missing_columns}")

# Filter HF dataset to matching question_pair_ids
filtered_hf_dataset = hf_dataset.filter(
    lambda example: example["question_pair_id"] in local_question_pair_ids or example["pair_id"] in local_pair_ids
)
print(f"Filtered HF dataset: {len(filtered_hf_dataset)}/{len(hf_dataset)} rows matched")

# 3. Create DataLoader with dynamic field handling
def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], str):
            collated[key] = values  # Keep strings as lists
        else:
            try:
                # Convert numerical values to tensors
                collated[key] = torch.tensor(values)
            except Exception:
                collated[key] = values  # Fallback for unsupported types
    return collated

dataloader = DataLoader(
    filtered_hf_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 4. Test the dataloader
if __name__ == "__main__":
    print("\nDataLoader test run:")
    print(f"Loaded {len(filtered_hf_dataset)} samples with matching question_pair_ids")
    print(f"Created dataloader with {len(dataloader)} batches")
    
    first_batch = next(iter(dataloader))
    print("\nFirst batch features:")
    for key, value in first_batch.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items (first: {value[0][:50] + '...' if isinstance(value[0], str) else value[0]})")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype} tensor of shape {value.shape}")
            if len(value) > 5:  # Truncate long outputs
                print(f"      Values[:5]: {value[:5]}")
        else:
            print(f"  {key}: {type(value).__name__} (first: {value[0]})")

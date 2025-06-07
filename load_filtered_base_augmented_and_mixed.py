import os
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch

# Configuration
root_path = "clone_detection_checkpoints_no_tokenize/fold_0/train/transform"
batch_size = 32
shuffle = True

# 1. Load local dataset
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

# 2. Load and filter Hugging Face dataset
hf_dataset = load_dataset("PoolC/5-fold-clone-detection-600k-5fold", split="train")

local_question_pair_ids = set()
local_pair_ids = set()
for example in local_dataset:
    if example["question_pair_id"] is not None:
        local_question_pair_ids.add(example["question_pair_id"])
    if example["pair_id"] is not None:
        local_pair_ids.add(example["pair_id"])
print(f"Found {len(local_question_pair_ids)} unique question_pair_ids in local dataset")
print(f"Found {len(local_pair_ids)} unique local_pair_ids in local dataset")

filtered_hf_dataset = hf_dataset.filter(
    lambda example: example["question_pair_id"] in local_question_pair_ids or example["pair_id"] in local_pair_ids
)
print(f"Filtered HF dataset: {len(filtered_hf_dataset)}/{len(hf_dataset)} rows matched")


# Create lookup dictionaries for HF dataset
hf_qid_map = {}
hf_pid_map = {}

for example in local_dataset:
    qid = example.get("question_pair_id")
    pid = example.get("pair_id")
    if qid is not None:
        hf_qid_map[qid] = example
    if pid is not None:
        hf_pid_map[pid] = example

# 3. Build combined dataset
combined_data = []

for local_ex in local_dataset:
    # Get matching HF example
    hf_ex = None
    if local_ex.get("question_pair_id") is not None:
        hf_ex = hf_qid_map.get(local_ex["question_pair_id"])
    elif local_ex.get("pair_id") is not None:
        hf_ex = hf_pid_map.get(local_ex["pair_id"])
    
    if hf_ex is None:
        continue
    
    # Create two new combined examples
    combined_data.append({
        "code1": local_ex["code1"],
        "code2": hf_ex["code2"],
        "similar": local_ex["similar"]
    })
    
    combined_data.append({
        "code1": hf_ex["code1"],
        "code2": local_ex["code2"],
        "similar": local_ex["similar"]
    })

# Convert to Hugging Face Dataset
combined_dataset = Dataset.from_list(combined_data)

combined_dataset = concatenate_datasets([filtered_hf_dataset, local_dataset, combined_dataset])

# 4. Create DataLoader
def collate_fn(batch):
    code1 = [item["code1"] for item in batch]
    code2 = [item["code2"] for item in batch]
    similar = torch.tensor([item["similar"] for item in batch], dtype=torch.long)
    return {
        "code1": code1,
        "code2": code2,
        "similar": similar
    }

dataloader = DataLoader(
    combined_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 5. Test the dataloader
if __name__ == "__main__":
    print(f"Created combined dataset with {len(combined_dataset)} samples")
    print(f"DataLoader batches: {len(dataloader)}")
    
    first_batch = next(iter(dataloader))
    print("\nFirst batch:")
    print(f"code1[0]: {first_batch['code1'][0][:100]}...")
    print(f"code2[0]: {first_batch['code2'][0][:100]}...")
    print(f"similar: {first_batch['similar'][:5]}...")
    print(f"Batch size: {len(first_batch['code1'])}")

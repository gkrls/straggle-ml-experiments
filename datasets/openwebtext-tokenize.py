#!/usr/bin/env python3
"""
Pre-tokenize OpenWebText parquet into flat memmap token files.
Run once:  python tokenize_owt.py --data /path/to/owt --out /path/to/owt/tokenized
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset

def tokenize_split(hf_split, tokenizer, out_path: Path):
    """Tokenize all docs, concatenate with EOS, write as uint16 memmap."""
    eos = tokenizer.eos_token_id
    # First pass: count total tokens so we can allocate the memmap
    print(f"  Counting tokens for {out_path.name} ...")
    total = 0
    for i, ex in enumerate(hf_split):
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        total += len(ids) + 1  # +1 for EOS separator
        if (i + 1) % 50000 == 0:
            print(f"    {i+1} docs, {total:,} tokens so far", flush=True)
    print(f"  Total: {len(hf_split)} docs, {total:,} tokens")

    # Second pass: write tokens
    mm = np.memmap(str(out_path), dtype=np.uint16, mode='w+', shape=(total,))
    offset = 0
    for i, ex in enumerate(hf_split):
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        mm[offset:offset + len(arr)] = arr
        offset += len(arr)
        if (i + 1) % 50000 == 0:
            print(f"    Written {i+1} docs ({offset:,} tokens)", flush=True)
    mm.flush()
    print(f"  Wrote {out_path} ({offset:,} tokens, {out_path.stat().st_size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Root of parquet dataset')
    parser.add_argument('--out', type=str, default=None, help='Output directory (default: <data>/tokenized)')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--val_fraction', type=float, default=0.0005)
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    out_dir = Path(args.out) if args.out else data_root / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)

    # # Reuse the existing layout resolver from the training script
    # sys.path.insert(0, str(Path(__file__).parent))
    # from gpt2 import hf_load_train_val_parquet
    # print(f"Loading parquet from {data_root} ...")
    # ds_train, ds_val = hf_load_train_val_parquet(data_root, val_fraction=args.val_fraction, seed=42)

    data_root = Path(args.data).resolve()
    train_glob = str(data_root / "train" / "*.parquet")
    val_glob = str(data_root / "val" / "*.parquet")

    print(f"Loading parquet from {data_root} ...")
    ds = load_dataset("parquet", data_files={"train": train_glob, "val": val_glob})
    ds_train = ds["train"]
    ds_val = ds["val"]


    print(f"\nTokenizing train split ({len(ds_train)} docs):")
    tokenize_split(ds_train, tokenizer, out_dir / "train.bin")

    print(f"\nTokenizing val split ({len(ds_val)} docs):")
    tokenize_split(ds_val, tokenizer, out_dir / "val.bin")

    # Write a small metadata file
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "train_tokens": int(np.memmap(str(out_dir / "train.bin"), dtype=np.uint16, mode='r').shape[0]),
        "val_tokens": int(np.memmap(str(out_dir / "val.bin"), dtype=np.uint16, mode='r').shape[0]),
    }
    import json
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Metadata: {meta}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Pre-tokenize OpenWebText parquet into flat memmap token files.
"""

import argparse, json
from pathlib import Path
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset


def tokenize_split(hf_split, tokenizer, out_path: Path, num_proc=1):
    eos = tokenizer.eos_token_id

    def tok_fn(example):
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        ids.append(eos)
        return {"ids": ids, "len": len(ids)}

    print(f"  Tokenizing {len(hf_split)} docs with {num_proc} processes ...", flush=True)
    mapped = hf_split.map(
        tok_fn,
        num_proc=num_proc,
        remove_columns=hf_split.column_names,
        desc=f"Tokenizing {out_path.stem}",
    )

    total = sum(mapped["len"])
    print(f"  {total:,} tokens total. Writing {out_path} ...", flush=True)

    mm = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=(total,))
    offset = 0
    for i, ids in enumerate(mapped["ids"]):
        arr = np.array(ids, dtype=np.uint16)
        mm[offset : offset + len(arr)] = arr
        offset += len(arr)
        if (i + 1) % 100_000 == 0:
            print(f"    {i+1} docs written ({offset:,} tokens)", flush=True)
    mm.flush()
    print(f"  Done: {out_path} ({total:,} tokens, {out_path.stat().st_size / 1e9:.2f} GB)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Tokenize OpenWebText parquet into flat uint16 memmap files")
    parser.add_argument("--data", type=str, required=True, help="Root dir containing train/ and val/ parquet subdirs")
    parser.add_argument("--out", type=str, default=None, help="Output dir (default: <data>/../tokenized)")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_proc", type=int, default=2, help="Number of parallel workers for tokenization")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    out_dir = Path(args.out).resolve() if args.out else data_root.parent / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 2**30  # suppress length warning

    train_glob = str(data_root / "train" / "*.parquet")
    val_glob = str(data_root / "val" / "*.parquet")

    print(f"Loading parquet from {data_root} ...", flush=True)
    ds = load_dataset("parquet", data_files={"train": train_glob, "val": val_glob})

    print(f"\n--- Train ({len(ds['train'])} docs) ---")
    train_tokens = tokenize_split(ds["train"].shuffle(seed=42), tokenizer, out_dir / "train.bin", num_proc=args.num_proc)

    print(f"\n--- Val ({len(ds['val'])} docs) ---")
    val_tokens = tokenize_split(ds["val"].shuffle(seed=42), tokenizer, out_dir / "val.bin", num_proc=args.num_proc)

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll done. Output in {out_dir}")
    print(f"  train.bin: {train_tokens:,} tokens")
    print(f"  val.bin:   {val_tokens:,} tokens")


if __name__ == "__main__":
    main()
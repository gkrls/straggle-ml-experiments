#!/usr/bin/env python3
import argparse, tarfile, re, random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

RE_MANY_NL = re.compile(r"\n{3,}")

def iter_texts(arc_dir: Path):
    """Stream .txt docs from Zenodo .tar.xz shards without extracting to disk."""
    for xz in sorted(arc_dir.glob("*.xz")):
        with tarfile.open(xz, mode="r|xz") as tar:  # streaming mode
            for m in tar:
                if not m.isfile() or not m.name.endswith(".txt"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                t = f.read().decode("utf-8", "ignore")
                yield RE_MANY_NL.sub("\n\n", t).strip()

def main():
    ap = argparse.ArgumentParser("Convert Zenodo OWT shards (.tar.xz) to Parquet")
    ap.add_argument("--arc_dir", required=True, help="Directory that contains *.xz shards (e.g. ~/datasets/openwebtext/openwebtext)")
    ap.add_argument("--out_dir", required=True, help="Output directory for Parquet files")
    ap.add_argument("--docs_per_shard", type=int, default=50_000, help="Documents per Parquet file")
    ap.add_argument("--val_frac", type=float, default=0.01, help="Validation fraction (0.0–1.0)")
    ap.add_argument("--seed", type=int, default=42, help="Split RNG seed")
    args = ap.parse_args()

    arc_dir = Path(args.arc_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not arc_dir.exists():
        raise FileNotFoundError(f"arc_dir not found: {arc_dir}")
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    schema = pa.schema([("text", pa.string())])
    rng = random.Random(args.seed)

    # per-split state
    batches = {"train": [], "val": []}
    writers = {"train": None, "val": None}
    idx = {"train": 0, "val": 0}

    def open_writer(split):
        path = out_dir / split / f"{split}-{idx[split]:05d}.parquet"
        writers[split] = pq.ParquetWriter(str(path), schema=schema, compression="zstd")

    def flush(split, rotate=False):
        """Write current batch for split. If rotate=True, close current file and open a new one."""
        if not batches[split]:
            return
        if writers[split] is None:
            open_writer(split)
        tbl = pa.Table.from_arrays([pa.array(batches[split])], schema=schema)
        writers[split].write_table(tbl)
        batches[split].clear()
        if rotate:
            writers[split].close()
            idx[split] += 1
            open_writer(split)

    docs_in_current_file = {"train": 0, "val": 0}

    for i, text in enumerate(iter_texts(arc_dir), 1):
        split = "val" if rng.random() < args.val_frac else "train"
        batches[split].append(text)
        docs_in_current_file[split] += 1

        # write & rotate when file reaches docs_per_shard
        if docs_in_current_file[split] >= args.docs_per_shard:
            flush(split, rotate=True)
            docs_in_current_file[split] = 0

        # write in chunks to avoid huge in-memory batches
        if len(batches[split]) >= min(args.docs_per_shard, 10_000):
            flush(split, rotate=False)

        if i % 100_000 == 0:
            print(f"[progress] processed {i:,} docs", flush=True)

    # flush tails and close writers
    for split in ("train", "val"):
        flush(split, rotate=False)
        if writers[split] is not None:
            writers[split].close()

    print("Done.")
    print(f"Train shards written under: {out_dir/'train'}")
    print(f"Val   shards written under: {out_dir/'val'}")

if __name__ == "__main__":
    main()

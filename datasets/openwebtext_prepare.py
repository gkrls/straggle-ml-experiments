#!/usr/bin/env python3
import argparse, os, tarfile, lzma, re
from pathlib import Path

NL_CLEAN = re.compile(r"\n\n\n+")
def clean(txt: str) -> str:
    # HF does similar: collapse 3+ newlines to 2
    return NL_CLEAN.sub("\n\n", txt).strip()

def iter_txts_from_arc(arc: Path):
    with lzma.open(arc, "rb") as comp:
        with tarfile.open(fileobj=comp, mode="r|*") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith(".txt"):
                    continue
                f = tar.extractfile(m)
                if f is None: 
                    continue
                yield clean(f.read().decode("utf-8", errors="ignore"))

def main():
    ap = argparse.ArgumentParser("Make big .txt shards from OpenWebText inner *_data.xz")
    ap.add_argument("--arc_dir", required=True, help="Directory with *_data.xz")
    ap.add_argument("--out_dir", required=True, help="Output dir for shard-*.txt")
    ap.add_argument("--shard_size_mb", type=int, default=1024)
    ap.add_argument("--max_arcs", type=int, default=None)
    args = ap.parse_args()

    arc_dir = Path(os.path.expanduser(args.arc_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_bytes = args.shard_size_mb * (1024**2)

    arcs = sorted([p for p in arc_dir.iterdir() if p.suffix == ".xz"])
    if args.max_arcs: arcs = arcs[:args.max_arcs]
    if not arcs: raise SystemExit(f"No *.xz found in {arc_dir}")

    shard_idx, bytes_written = 0, 0
    f = (out_dir / f"shard-{shard_idx:05d}.txt").open("w", encoding="utf-8")
    docs = 0
    for i, arc in enumerate(arcs, 1):
        print(f"[{i}/{len(arcs)}] {arc.name}")
        for txt in iter_txts_from_arc(arc):
            b = txt.encode("utf-8")
            # roll shard if we'd exceed target size
            if bytes_written and bytes_written + len(b) + 2 > shard_bytes:
                f.close()
                shard_idx += 1
                f = (out_dir / f"shard-{shard_idx:05d}.txt").open("w", encoding="utf-8")
                bytes_written = 0
            f.write(txt); f.write("\n\n")
            bytes_written += len(b) + 2
            docs += 1
    f.close()
    print(f"[done] wrote shards to {out_dir}  (docs: {docs:,})")

if __name__ == "__main__":
    main()

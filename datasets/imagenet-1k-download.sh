#!/usr/bin/env bash
# Usage: ./get_imagenet.sh /path/to/dir
# Env knobs:
#   N_PARTS=16        # parallel connections for aria2 / wget chunking
#   SPLIT_SIZE_MIN=1M # aria2c chunk size hint

set -euo pipefail

TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEVKIT_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

TARGET_DIR="${1:-.}"
mkdir -p "$TARGET_DIR"

N_PARTS="${N_PARTS:-16}"
SPLIT_SIZE_MIN="${SPLIT_SIZE_MIN:-1M}"

have() { command -v "$1" >/dev/null 2>&1; }

download_aria() {
  local url="$1" out="$2"
  echo "[aria2c] $out"
  # -x/-s = connections, -k = min split size, -c = resume, -d/-o set dest
  aria2c -c -x"$N_PARTS" -s"$N_PARTS" -k"$SPLIT_SIZE_MIN" \
         -d "$(dirname "$out")" -o "$(basename "$out")" \
         "$url"
}

# Get headers with wget only; extract Accept-Ranges and Content-Length
probe_with_wget() {
  local url="$1"
  wget --spider --server-response -O - "$url" 2>&1 \
  | awk 'BEGIN{IGNORECASE=1}
         tolower($1) ~ /^accept-ranges:/  {ar=$2}
         tolower($1) ~ /^content-length:/ {cl=$2}
         END{ if(!ar) ar=""; if(!cl) cl=0; printf "%s %s\n", ar, cl }'
}

download_wget_parallel() {
  local url="$1" out="$2" nparts="$3"
  echo "[wget-parallel] Probing range support for $url"
  read -r accept_ranges content_length < <(probe_with_wget "$url")

  if [[ "$accept_ranges" != "bytes" || "$content_length" -le 0 ]]; then
    echo "[wget] Server does not advertise range support (or size unknown). Using single stream."
    wget -c --show-progress -O "$out" "$url"
    return
  fi

  echo "[wget-parallel] Accept-Ranges: bytes, size: $content_length bytes, parts: $nparts"
  local chunk=$(( (content_length + nparts - 1) / nparts ))
  local -a pids=()

  for ((i=0; i<nparts; i++)); do
    local start=$(( i * chunk ))
    [[ $start -ge $content_length ]] && break
    local end=$(( start + chunk - 1 ))
    [[ $end -ge $content_length ]] && end=$(( content_length - 1 ))
    local part="$(printf "%s.part.%03d" "$out" "$i")"

    echo "[wget-parallel] Part $i bytes=${start}-${end} -> $part"
    # -c resumes each part if re-run; --show-progress prints a bar per part (lines will interleave)
    wget -c --show-progress \
         --header="Range: bytes=${start}-${end}" \
         -O "$part" "$url" &
    pids+=( "$!" )
  done

  # Wait for all parts
  set +e
  err=0
  for pid in "${pids[@]}"; do
    wait "$pid" || err=1
  done
  set -e
  if [[ $err -ne 0 ]]; then
    echo "One or more part downloads failed. Re-run the script to resume." >&2
    exit 1
  fi

  # Stitch parts in order
  echo "[wget-parallel] Assembling parts -> $out"
  cat "$out".part.* > "${out}.tmp"
  mv "${out}.tmp" "$out"
  rm -f "$out".part.*
}

download() {
  local url="$1" out="$2"
  if have aria2c; then
    download_aria "$url" "$out"
  elif have wget; then
    download_wget_parallel "$url" "$out" "$N_PARTS"
  else
    echo "Neither aria2c nor wget is available." >&2
    exit 1
  fi
}

download "$TRAIN_URL" "$TARGET_DIR/ILSVRC2012_img_train.tar"
download "$VAL_URL" "$TARGET_DIR/ILSVRC2012_img_val.tar"
download "$DEVKIT_URL" "$TARGET_DIR/ILSVRC2012_devkit_t12.tar.gz"

echo "All done."

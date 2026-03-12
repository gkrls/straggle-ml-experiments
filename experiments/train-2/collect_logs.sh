set -euo pipefail

HOSTS=("hpdc-gnode1" "hpdc-gnode2" "hpdc-gnode3" "hpdc-gnode4" "hpdc-gnode5" "hpdc-gnode6")
RANKS=($(seq 0 $((${#HOSTS[@]} - 1))))

REMOTE_JSON=""
OUT_DIR=""
IGNORE=""
PREFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2}_"; shift 2 ;;
    --ignore) IGNORE="$2"; shift 2 ;;
    -o)    OUT_DIR="$2"; shift 2 ;;
    -f)   REMOTE_JSON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$REMOTE_JSON" ]]; then
  echo "Error: -f <remote_json_path> is required"
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="collected_logs_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"

IFS=',' read -ra SKIP <<< "$IGNORE"

for rank in "${RANKS[@]}"; do
  skip=0
  for s in "${SKIP[@]}"; do
    [[ "$s" == "$rank" ]] && skip=1
  done
  if [[ $skip -eq 1 ]]; then
    echo "[rank $rank] SKIPPED (${HOSTS[$rank]})"
    continue
  fi

  host="${HOSTS[$rank]}"
  dest="$OUT_DIR/${PREFIX}rank${rank}.json"
  echo -n "[rank $rank] $host:$REMOTE_JSON -> $dest ... "
  if scp -q "$host:$REMOTE_JSON" "$dest" 2>/dev/null; then
    echo "OK"
  else
    echo "FAILED"
  fi
done

echo ""
echo "Collected logs in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.json 2>/dev/null || echo "  (none)"
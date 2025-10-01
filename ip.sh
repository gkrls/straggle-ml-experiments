RANK="${RANK:-${1:-}}"
if [[ -z "${RANK}" ]]; then
  echo "Usage: RANK=<num> $0 [rank]   (DEV can override interface, default: ${DEV})" >&2
  exit 1
fi

octet=$(( RANK + 1 ))
sudo ip -4 addr replace 42.0.0.${octet}/24 dev ens4f0
sudo ip -4 addr replace 42.0.1.${octet}/24 dev ens4f1
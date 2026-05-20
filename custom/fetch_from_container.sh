#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-coroni}"
CONTAINER_ID="${CONTAINER_ID:-2c35433c4d8d}"
DESKTOP_DIR="${DESKTOP_DIR:-$HOME/Desktop}"

usage() {
  echo "Usage: $0 vortex/path/to/file [vortex/path/to/other-file ...]"
  echo
  echo "Example:"
  echo "  $0 vortex/build/run1_ip.log vortex/build/run2_ip.log"
  echo
  echo "Optional overrides:"
  echo "  SSH_HOST=coroni CONTAINER_ID=2c35433c4d8d DESKTOP_DIR=\$HOME/Desktop $0 vortex/build/run1_ip.log"
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d "$DESKTOP_DIR" ]]; then
  echo "error: desktop directory does not exist: $DESKTOP_DIR" >&2
  exit 1
fi

for input_path in "$@"; do
  case "$input_path" in
    vortex/*)
      container_path="/$input_path"
      ;;
    /vortex/*)
      container_path="$input_path"
      ;;
    *)
      echo "error: path must start with 'vortex/' or '/vortex/': $input_path" >&2
      usage >&2
      exit 2
      ;;
  esac

  filename="$(basename "$container_path")"
  remote_tmp="/tmp/${CONTAINER_ID}_${BASHPID}_${filename}"
  local_dest="$DESKTOP_DIR/$filename"

  echo "Copying $CONTAINER_ID:$container_path from $SSH_HOST to $local_dest"

  ssh "$SSH_HOST" "docker cp '$CONTAINER_ID:$container_path' '$remote_tmp'"
  scp "$SSH_HOST:$remote_tmp" "$local_dest"
  ssh "$SSH_HOST" "rm -f '$remote_tmp'"

  echo "Saved to $local_dest"
done

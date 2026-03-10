#!/usr/bin/env bash
set -euo pipefail

# Run this script from anywhere inside the repo.
cd "$(git rev-parse --show-toplevel)"

# List of files to untrack (keep locally as untracked)
FILES=(
  ".clang-format2"
  "chat.rapidgpt"
  "dodo.sh"
  "lab2-1.diff"
  "lab2-2.diff"
  "lab3_repo.zip"
  "priority.diff"
  "run.log"
  "tcu.diff"
  "tensor1.diff"
  "tensor2.diff"
  "tensor3.diff"
  "tensor4.diff"
  "tensor5.diff"
  "tensor6.diff"
  "tensor7.diff"
  "tests/regression/sgemm_tcu/fast_dotp.cpp"
  "tests/regression/sgemm_tcu/tensor_cfg.py"
  "tests/regression/sgemm_tcu/tensor_i4.cpp"
  "tests/riscv/isa/ramulator.stats.log"
  "tests/riscv/isa/trace/ramulator.log.ch0"
  "tests/riscv/isa/trace/ramulator.log.ch1"
)

echo "[1/3] Restoring any missing working-tree copies (if needed)…"
for f in "${FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    # Try to restore the file content to your working tree (no staging).
    git restore --worktree --source=HEAD -- "$f" 2>/dev/null || true
  fi
done

echo "[2/3] Stopping tracking of the listed files (index only)…"
changed=0
for f in "${FILES[@]}"; do
  if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    git rm --cached -- "$f" >/dev/null
    echo "  untracked: $f"
    changed=1
  else
    echo "  (not tracked): $f"
  fi
done

echo "[3/3] Committing and pushing (if anything changed)…"
if [[ "$changed" -eq 1 ]]; then
  git commit -m "Stop tracking accidentally committed files; keep local copies untracked"
  git push
  echo "Done. Remote cleaned; local files remain on disk as untracked."
else
  echo "Nothing to commit. (All listed files were already untracked.)"
fi
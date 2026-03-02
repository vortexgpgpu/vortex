#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TOOLDIR=${TOOLDIR:-/opt}
DRIVERS=${DRIVERS:-"simx rtlsim"}
CORES_LIST=${CORES_LIST:-"1 4"}
WARPS_LIST=${WARPS_LIST:-"4 8"}
THREADS_LIST=${THREADS_LIST:-"4 8"}
N_LIST=${N_LIST:-"16 32 64"}
DXA_OPTS_BASE=${DXA_OPTS_BASE:-"-t4 -c4 -m2"}
VERIFY=${VERIFY:-1}
COMMON_CONFIGS=${COMMON_CONFIGS:-"-DEXT_DXA_ENABLE"}
CASE_TIMEOUT_SEC=${CASE_TIMEOUT_SEC:-0}
TIMEOUT_KILL_SEC=${TIMEOUT_KILL_SEC:-10}

OUT_CSV=${OUT_CSV:-"perf/results/sgemm2_dxa_compare_full.csv"}
LOG=${LOG:-"/tmp/sgemm2_dxa_compare_all.log"}

if [[ -d "$TOOLDIR/verilator/bin" ]]; then
  export PATH="$TOOLDIR/verilator/bin:$PATH"
fi

if [[ "$OUT_CSV" != /* ]]; then
  OUT_CSV="$ROOT_DIR/$OUT_CSV"
fi
if [[ "$LOG" != /* ]]; then
  LOG="$ROOT_DIR/$LOG"
fi

mkdir -p "$(dirname "$OUT_CSV")"
mkdir -p "$(dirname "$LOG")"

header="driver,cores,warps,threads,n,timeout_sec,base_status,base_instrs,base_cycles,base_ipc,dxa_status,dxa_instrs,dxa_cycles,dxa_ipc,speedup_base_over_dxa"
if [[ ! -f "$OUT_CSV" ]]; then
  echo "$header" > "$OUT_CSV"
fi

declare -A DONE
while IFS=',' read -r driver cores warps threads n _rest; do
  if [[ "$driver" == "driver" || -z "$driver" ]]; then
    continue
  fi
  DONE["${driver}_${cores}_${warps}_${threads}_${n}"]=1
done < "$OUT_CSV"

pick_timeout_sec() {
  local driver=$1
  local n=$2
  local cores=$3
  local warps=$4
  local threads=$5

  if [[ "$CASE_TIMEOUT_SEC" -gt 0 ]]; then
    echo "$CASE_TIMEOUT_SEC"
    return
  fi

  local base
  if [[ "$driver" == "simx" ]]; then
    if (( n <= 16 )); then
      base=30
    elif (( n <= 32 )); then
      base=45
    elif (( n <= 64 )); then
      base=90
    elif (( n <= 128 )); then
      base=180
    else
      base=300
    fi
  else
    if (( n <= 16 )); then
      base=60
    elif (( n <= 32 )); then
      base=120
    elif (( n <= 64 )); then
      base=300
    elif (( n <= 128 )); then
      base=900
    else
      base=1800
    fi
  fi

  local load_factor=$(( cores * warps * threads ))
  if (( load_factor >= 256 )); then
    base=$(( base * 2 ))
  elif (( load_factor >= 128 )); then
    base=$(( (base * 3) / 2 ))
  fi
  echo "$base"
}

run_case() {
  local driver=$1
  local app=$2
  local cores=$3
  local warps=$4
  local threads=$5
  local args=$6

  local timeout_sec
  timeout_sec=$(pick_timeout_sec "$driver" "$RUN_N" "$cores" "$warps" "$threads")

  local cmd="./ci/blackbox.sh --driver=${driver} --app=${app} --cores=${cores} --warps=${warps} --threads=${threads} --args=${args}"
  local out
  if ! out=$(cd "$ROOT_DIR" \
      && CONFIGS="$COMMON_CONFIGS" TOOLDIR="$TOOLDIR" \
      timeout --signal=TERM --kill-after="${TIMEOUT_KILL_SEC}s" "${timeout_sec}s" \
      ./ci/blackbox.sh \
      --driver="$driver" --app="$app" \
      --cores="$cores" --warps="$warps" --threads="$threads" \
      --args="$args" 2>&1); then
    local status=$?
    echo "$out" >> "$LOG"
    if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
      echo "TIMEOUT: driver=$driver app=$app c=$cores w=$warps t=$threads n=$RUN_N timeout=${timeout_sec}s" | tee -a "$LOG"
      echo "timeout NA NA NA $timeout_sec"
      return 0
    fi
    echo "$out" | tee -a "$LOG"
    echo "ERROR: run failed for driver=$driver app=$app c=$cores w=$warps t=$threads n=$RUN_N args='$args' status=$status" | tee -a "$LOG"
    echo "error NA NA NA $timeout_sec"
    return 0
  fi

  echo "$out" >> "$LOG"

  local perf
  perf=$(echo "$out" | awk '/^PERF: / {line=$0} END {print line}')
  if [[ -z "$perf" ]]; then
    echo "ERROR: PERF line missing for driver=$driver app=$app c=$cores w=$warps t=$threads n=$RUN_N args='$args'" | tee -a "$LOG"
    echo "noperf NA NA NA $timeout_sec"
    return 0
  fi

  local instrs cycles ipc
  instrs=$(echo "$perf" | awk -F'[=, ]+' '{print $3}')
  cycles=$(echo "$perf" | awk -F'[=, ]+' '{print $5}')
  ipc=$(echo "$perf" | awk -F'[=, ]+' '{print $7}')
  echo "ok $instrs $cycles $ipc $timeout_sec"
}

echo "== sgemm2 vs sgemm2_dxa full compare ==" | tee -a "$LOG"
echo "drivers=$DRIVERS cores=$CORES_LIST warps=$WARPS_LIST threads=$THREADS_LIST n=$N_LIST timeout=${CASE_TIMEOUT_SEC}s(auto when 0)" | tee -a "$LOG"

for driver in $DRIVERS; do
  for cores in $CORES_LIST; do
    for warps in $WARPS_LIST; do
      for threads in $THREADS_LIST; do
        for n in $N_LIST; do
          RUN_N=$n
          key="${driver}_${cores}_${warps}_${threads}_${n}"
          if [[ -n "${DONE[$key]:-}" ]]; then
            echo "[skip] $key already exists in $OUT_CSV" | tee -a "$LOG"
            continue
          fi

          base_args="-n${n}"
          dxa_args="-n${n} ${DXA_OPTS_BASE}"
          if [[ "$VERIFY" -eq 0 ]]; then
            base_args="${base_args} -q"
            dxa_args="${dxa_args} -q"
          fi

          echo "[run] driver=$driver c=$cores w=$warps t=$threads n=$n" | tee -a "$LOG"
          base_metrics=$(run_case "$driver" sgemm2 "$cores" "$warps" "$threads" "$base_args")
          read -r b_status b_instrs b_cycles b_ipc timeout_sec <<< "$base_metrics"

          dxa_metrics=$(run_case "$driver" sgemm2_dxa "$cores" "$warps" "$threads" "$dxa_args")
          read -r d_status d_instrs d_cycles d_ipc timeout_sec2 <<< "$dxa_metrics"

          if [[ -n "${timeout_sec2:-}" && "${timeout_sec2}" != "NA" ]]; then
            timeout_sec=$timeout_sec2
          fi

          if [[ "$b_status" != "ok" || "$d_status" != "ok" || -z "$d_cycles" || "$d_cycles" == "0" || "$d_cycles" == "NA" ]]; then
            speedup="nan"
          else
            speedup=$(awk -v b="$b_cycles" -v d="$d_cycles" 'BEGIN {printf "%.6f", b / d}')
          fi

          echo "${driver},${cores},${warps},${threads},${n},${timeout_sec},${b_status},${b_instrs},${b_cycles},${b_ipc},${d_status},${d_instrs},${d_cycles},${d_ipc},${speedup}" >> "$OUT_CSV"
          DONE["$key"]=1
        done
      done
    done
  done
done

echo "DONE. Results: $OUT_CSV"

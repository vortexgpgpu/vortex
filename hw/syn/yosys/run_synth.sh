#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Yosys wrapper: generates synth.ys and runs it
#
# Usage (env):
#   TOP=<top> SRC_FILE=<filelist.f>
#   [LIB_TGT=<tech.lib>]
#   [LIB_ROOT=<dir_with_libs>]
#   [SDC_FILE=<constraints.sdc>]
#   [OUT_DIR=out]
#   [RPT_DIR=reports]
#   [RUN_SYNTH=1]
#   [RUN_MAP=1]
#   [RUN_STA=0]
#   [CLOCK_FREQ=800]
#   [DELAY_UNC=0.02]
#   [DELAY_IO=0.05]
#   [DFF_DONT_USE="<cell_glob> ..."]
#   [ABC_DRIVER_CELL=<lib_cell>] [ABC_LOAD=<ff>]
#   [YOSYS_FLATTEN=1]
#   [YOSYS_SHARE=1]
#   [SAIF_FILE=<activity.saif>]
#   [SAIF_INST=<tb.dut>]
#   [BB_MODULES="modA,modB"]
#   [PDK=asap7]
#   [SRAM_BIT_AREA=0.1]
#   [SRAM_OVERHEAD=100.0]
#   [SRAM_W_PORTS="wdata,rdata"]
#   [SRAM_A_PORTS="addr,waddr,raddr"]
#   [YOSYS=<absolute path to yosys>]    # default: PATH lookup
#   [STA=<absolute path to sta>]        # default: PATH lookup
#   ./run_synth.sh
#
set -euo pipefail

# Tool binaries: prefer caller-supplied env vars (set by hw/syn/common.mk
# to absolute $(TOOLDIR)/{yosys,sta}/bin/...) so the build is self-
# contained. Fall back to PATH lookup for callers that don't set them.
YOSYS="${YOSYS:-yosys}"
STA="${STA:-sta}"

die() { echo "FATAL: $*" >&2; exit 1; }
log() { echo "[run_synth] $*"; }

now_ms() { date +%s%3N; }
hhmmss() { local ms=$1; local s=$((ms/1000)); printf "%02d:%02d:%02d" $((s/3600)) $(((s/60)%60)) $((s%60)); }
stamp() { local label="$1"; local now=$(now_ms); printf "TIME %-16s  stage=%s  total=%s\n" "$label" "$(hhmmss $((now-LAP_START)))" "$(hhmmss $((now-START_MS)))"; LAP_START=$now; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- config --------
TOP="${TOP:-}"; [[ -n "${TOP}" ]] || die "TOP required"
SRC_FILE="${SRC_FILE:-}"; [[ -n "${SRC_FILE}" ]] || die "SRC_FILE required"

LIB_ROOT="${LIB_ROOT:-}"
LIB_TGT="${LIB_TGT:-}"
PDK="${PDK:-custom}"
SDC_FILE="${SDC_FILE:-}"
OUT_DIR="${OUT_DIR:-out}"
RPT_DIR="${RPT_DIR:-reports}"
RUN_SYNTH="${RUN_SYNTH:-1}"
RUN_MAP="${RUN_MAP:-1}"
RUN_STA="${RUN_STA:-0}"
CLOCK_FREQ="${CLOCK_FREQ:-800}"
DELAY_UNC="${DELAY_UNC:-0.02}"
DELAY_IO="${DELAY_IO:-0.05}"
DFF_DONT_USE="${DFF_DONT_USE:-}"
ABC_DRIVER_CELL="${ABC_DRIVER_CELL:-}"
ABC_LOAD="${ABC_LOAD:-}"
YOSYS_FLATTEN="${YOSYS_FLATTEN:-1}"
YOSYS_SHARE="${YOSYS_SHARE:-1}"
SAIF_FILE="${SAIF_FILE:-}"
SAIF_INST="${SAIF_INST:-}"
BB_MODULES="${BB_MODULES:-}"

# Area Estimation Defaults
SRAM_BIT_AREA="${SRAM_BIT_AREA:-0.1}"
SRAM_OVERHEAD="${SRAM_OVERHEAD:-100.0}"
SRAM_W_PORTS="wdata,rdata"
SRAM_A_PORTS="addr,waddr,raddr"

mkdir -p "$OUT_DIR" "$RPT_DIR"

YS="$OUT_DIR/synth.ys"
YLOG="$RPT_DIR/yosys.log"
NET_PRE="$OUT_DIR/${TOP}_syn.v"
NET_POST="$OUT_DIR/${TOP}_mapped.v"
JOUT="$OUT_DIR/${TOP}.json"

START_MS=$(now_ms); LAP_START=$START_MS

# -------- parse VCS-style filelist --------
declare -a SRC_FILES=()
declare -a INC_DIRS=()
declare -a DEFINES=()

expand_filelist() {
  local f="$1"
  [[ -f "$f" ]] || die "filelist not found: $f"
  local base
  base="$(cd "$(dirname "$f")" && pwd)"
  local next_is_file=0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    local line="${raw#"${raw%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    for tok in $line; do
      if [[ "$tok" == "-f" ]]; then next_is_file=1; continue; fi
      if [[ $next_is_file -eq 1 ]]; then
        [[ "$tok" == /* ]] || tok="$base/$tok"
        expand_filelist "$tok"; next_is_file=0; continue
      fi
      if [[ "$tok" == +incdir+* ]]; then
        IFS='+' read -r -a dirs <<< "${tok#+incdir+}"
        for d in "${dirs[@]}"; do
          [[ "$d" == /* ]] || d="$base/$d"
          INC_DIRS+=("$d")
        done
        continue
      fi
      if [[ "$tok" == +define+* ]]; then d="${tok#+define+}"; [[ "$d" == *=* ]] && DEFINES+=("$d") || DEFINES+=("$d=1"); continue; fi
      [[ "$tok" == +* ]] && continue
      [[ "$tok" == /* ]] || tok="$base/$tok"
      SRC_FILES+=("$tok")
    done
  done < "$f"
}
expand_filelist "$SRC_FILE"

# -------- liberty discovery --------
mapfile -t LIB_LIST < <( ( [[ -n "$LIB_ROOT" ]] && find "$LIB_ROOT" -type f -name '*.lib' -print ) | sort -u )
[[ -n "$LIB_TGT" ]] && LIB_LIST+=("$LIB_TGT")
mapfile -t LIB_LIST < <(printf '%s\n' "${LIB_LIST[@]}" | awk 'NF && !seen[$0]++')
[[ ${#LIB_LIST[@]} -gt 0 ]] || die "no Liberty libraries selected"
for lib in "${LIB_LIST[@]}"; do
  [[ -f "$lib" ]] || die "Liberty file not found: $lib"
done
if [[ "$RUN_MAP" == "1" ]]; then
  [[ -n "$LIB_TGT" ]] || die "LIB_TGT is required for technology mapping"
  [[ -f "$LIB_TGT" ]] || die "LIB_TGT not found: $LIB_TGT"
fi

# -------- compute PERIOD_NS, TARGET_UNC, and TARGET_IO --------
PERIOD_NS="$(python3 - <<PY
mhz=float("$CLOCK_FREQ")
print(1000.0/mhz)
PY
)"

TARGET_UNC="$(python3 - <<PY
p=float("$PERIOD_NS")
print(p*float("$DELAY_UNC"))
PY
)"

TARGET_IO="$(python3 - <<PY
p=float("$PERIOD_NS")
print(p*float("$DELAY_IO"))
PY
)"

ABC_PERIOD="$(python3 - <<PY
print(float("$PERIOD_NS") * 1000.0)
PY
)"

RESOLVED_SDC=""
ABC_CONSTR=""
if [[ -n "$SDC_FILE" && -f "$SDC_FILE" ]]; then
  RESOLVED_SDC="$OUT_DIR/${TOP}.resolved.sdc"
  {
    echo "# Auto-generated by run_synth.sh"
    echo "set target_period      $PERIOD_NS"
    echo "set target_uncertainty $TARGET_UNC"
    echo "set target_io_delay    $TARGET_IO"
    echo ""
    cat "$SDC_FILE"
  } > "$RESOLVED_SDC"
fi

if [[ "$RUN_MAP" == "1" && -n "$ABC_DRIVER_CELL" ]]; then
  ABC_CONSTR="$OUT_DIR/${TOP}.abc.constr"
  {
    printf "set_driving_cell %s\n" "$ABC_DRIVER_CELL"
    printf "set_load %s\n" "${ABC_LOAD:-0.0}"
  } > "$ABC_CONSTR"
fi

if [[ "$RUN_MAP" == "1" ]]; then
  # Bounded ABC mapping script -- see the comment at the `abc` command below
  # for why the default script's sequential passes are excluded.
  cat > "$OUT_DIR/abc_map.script" <<'ABCEOF'
strash
&get -n
&nf {D}
&put
buffer
upsize {D}
dnsize {D}
stime -p
ABCEOF
fi

log "TOP=$TOP  RUN_SYNTH=$RUN_SYNTH RUN_MAP=$RUN_MAP RUN_STA=$RUN_STA"
log "PDK=$PDK  Liberty=${#LIB_LIST[@]}"
log "Sources=${#SRC_FILES[@]}  Incdirs=${#INC_DIRS[@]}  Defines=${#DEFINES[@]}"
[[ -n "$ABC_PERIOD" ]] && log "ABC_PERIOD=$ABC_PERIOD ps" || log "ABC_PERIOD not set"
[[ -n "$RESOLVED_SDC" ]] && log "Resolved SDC: $RESOLVED_SDC"
[[ -n "$ABC_CONSTR" ]] && log "ABC constraints: $ABC_CONSTR"
[[ -n "$SAIF_FILE" ]] && log "SAIF_FILE=$SAIF_FILE  SAIF_INST=${SAIF_INST:-<none>}"

# -------- synth.ys --------
log "Writing $YS"
{
  echo "# Auto-generated by run_synth.sh"
  echo "verilog_defaults -add -sv"
  for d in "${INC_DIRS[@]}"; do printf "verilog_defaults -add -I %q\n" "$d"; done
  for d in "${DEFINES[@]}"; do printf "verilog_defaults -add -D %q\n" "$d"; done

  for l in "${LIB_LIST[@]}"; do printf "read_liberty -lib %q\n" "$l"; done
  echo "# read sources"
  for s in "${SRC_FILES[@]}"; do printf "read_verilog -defer %q\n" "$s"; done

  if [[ -n "$BB_MODULES" ]]; then
    IFS=',' read -r -a bbmods <<< "$BB_MODULES"
    for m in "${bbmods[@]}"; do printf "blackbox %q\n" "$m"; done
  fi

  printf "hierarchy -check -top %q\n" "$TOP"
  if [[ "$RUN_SYNTH" == "1" ]]; then
    echo "proc; opt"
    echo "fsm; opt"
    echo "memory; opt"
    echo "memory_map; opt"
    echo "alumacc; wreduce; opt"
    if [[ "$YOSYS_SHARE" == "1" ]]; then
      echo "share; opt"
    fi
    if [[ "$YOSYS_FLATTEN" == "1" ]]; then
      echo "flatten; opt"
    fi
    echo "techmap; opt"
  fi

  printf "write_verilog -noattr -noexpr -renameprefix syn_ %q\n" "$NET_PRE"
  printf "write_json %q\n" "$JOUT"

  if [[ "$RUN_MAP" == "1" ]]; then
    # Flop cell selection is the single largest area term (flops are ~32% of
    # this design), and dfflibmap picks one cell per FF type by its own cost
    # model. DFF_DONT_USE withholds cells whose selection costs area for no
    # timing: ASAP7 carries a non-inverting flop only at x4 drive, so yosys
    # 0.69 -- which prefers non-inverting cells where 0.57 took the cheapest --
    # mapped all 129564 flops to DFFHQx4 (0.3645) instead of DFFHQNx1 (0.2916),
    # +25% sequential area with Fmax slightly WORSE.
    #
    # %s, not %q: these are globs, and %q escapes the asterisks
    # (\*DFFHQx4\*), which yosys then matches literally and never applies.
    # Values are whitespace-split above, so no token can carry a space.
    dff_dont_use=""
    for c in $DFF_DONT_USE; do
      dff_dont_use+=" -dont_use $c"
    done
    printf "dfflibmap%s -liberty %q\n" "$dff_dont_use" "$LIB_TGT"
    # Bounded ABC script: yosys's default -liberty script minus its sequential
    # passes (&fraig; scorr; dc2; dretime; retime; &dch). Two reasons, both
    # load-bearing:
    #
    #   - scorr/&dch have no runtime bound and blow up on flop count: gfx
    #     (517914 DFFs) sat in them for 16+ hours without terminating, while
    #     this script maps the identical netlist in under 4 minutes and meets
    #     the same 2500ps target (1469ps critical path). core and rtu owed
    #     their 4-5 hour builds to the same passes.
    #   - retime/dretime move flops, so the gate would be measuring a design
    #     ABC re-architected rather than the RTL as written. A synthesis
    #     regression gate must not do sequential re-timing.
    #
    # The mapper (&nf) is unchanged from the default script, so results stay
    # comparable in kind; baselines were re-recorded when this landed.
    #
    # The script goes through a FILE, not the +cmd;cmd inline form: the inline
    # form's ;&{} characters have to survive printf %q, this .ys file AND
    # yosys's tokenizer, and in practice they did not (%q's backslashes reached
    # ABC verbatim, and yosys's brace handling relocated the first {D}'s
    # closing brace). A file path has no metacharacters to lose. Yosys applies
    # the {D} -> "-D <period>" substitution to script files all the same.
    if [[ -n "$ABC_CONSTR" ]]; then
      printf "abc -markgroups -D %q -liberty %q -constr %q -script %q\n" "$ABC_PERIOD" "$LIB_TGT" "$ABC_CONSTR" "$OUT_DIR/abc_map.script"
    else
      printf "abc -markgroups -D %q -liberty %q -script %q\n" "$ABC_PERIOD" "$LIB_TGT" "$OUT_DIR/abc_map.script"
    fi
    # Drop what mapping left behind before anything measures or reads the
    # netlist. Without it the gate reports dead cells as area, and the written
    # netlist carries ~129 undriven `assign w = 'hx` wires -- sv2v function
    # scopes that survive as debris. OpenSTA's Verilog reader rejects those
    # outright once Yosys marks them `signed` (0.69 does, 0.57 did not), which
    # is how a missing clean read as a Yosys incompatibility.
    printf "opt_clean -purge\n"
    printf "tee -o %q stat -liberty %q -top %q -width -tech cmos\n" "$RPT_DIR/stat_lib.rpt" "$LIB_TGT" "$TOP"
    printf "write_verilog -noattr -noexpr %q\n" "$NET_POST"
  fi
} > "$YS"
stamp "gen-ys"

# -------- run yosys --------
log "$YOSYS -q -s $YS -l $YLOG"
"$YOSYS" -q -s "$YS" -l "$YLOG"
stamp "yosys"

# -------- strip net signedness for OpenSTA --------
# OpenSTA's Verilog grammar has no `signed` keyword, so ONE signed net
# declaration makes it reject the whole netlist:
#   Error: 171 <netlist> line N, syntax error
# and the DUT reports no timing or power at all. Yosys preserves signedness on
# every wire it writes as of 0.69; 0.57 dropped it, which is why this surfaced
# with the toolchain bump rather than with the RTL.
#
# Signedness is information-free in a MAPPED netlist -- these are bit vectors
# between library cells, and nothing in timing or power interprets them -- so
# removing the qualifier cannot change what STA computes.
#
# This is a WORKAROUND, not a fix (AGENTS.md S3): the defect is OpenSTA's
# reader, which should accept and ignore `signed`. Follow-up is an upstream
# patch there, after which this block goes away.
if [[ "$RUN_MAP" == "1" && -f "$NET_POST" ]]; then
  n=$(grep -cE '^[[:space:]]*(wire|reg|input|output|inout)[[:space:]]+signed[[:space:]]' "$NET_POST" || true)
  if [[ "$n" != "0" ]]; then
    sed -i -E 's/^([[:space:]]*(wire|reg|input|output|inout))[[:space:]]+signed[[:space:]]+/\1 /' "$NET_POST"
    log "stripped 'signed' from $n net declarations (OpenSTA reader limitation)"
  fi
  stamp "strip-signed"
fi

# -------- run sram area estimation --------
if [[ -n "$BB_MODULES" ]]; then
  if [[ -f "$JOUT" && -f "$SCRIPT_DIR/sram_cost.py" ]]; then
      log "Running SRAM Area Estimator..."
      BB_ARGS=$(echo "$BB_MODULES" | tr ',' ' ')

      python3 "$SCRIPT_DIR/sram_cost.py" "$JOUT" \
          --top "$TOP" \
          --modules $BB_ARGS \
          --width-ports $SRAM_W_PORTS \
          --addr-ports $SRAM_A_PORTS \
          --bit-area "$SRAM_BIT_AREA" \
          --overhead "$SRAM_OVERHEAD" \
          | tee "$RPT_DIR/sram_area.rpt"
  else
      log "Warning: Skipping SRAM estimation. (Missing JSON or script)"
  fi
else
  log "Skipping SRAM Area Estimation (BB_MODULES is empty)"
fi

# -------- optional OpenSTA (run_sta.tcl colocated) --------
if [[ "$RUN_STA" == "1" ]]; then
  # Resolve $STA: prefer the absolute path the caller gave us; otherwise
  # accept any sta on PATH; otherwise error out.
  if [[ ! -x "$STA" ]] && ! command -v "$STA" >/dev/null 2>&1; then
    echo "ERROR: OpenSTA ('$STA') not found; required when RUN_STA=1" >&2
    exit 1
  fi
  STA_SCRIPT="$SCRIPT_DIR/run_sta.tcl"
  NETLIST="$NET_POST"; [[ -f "$NETLIST" ]] || NETLIST="$NET_PRE"
  STA_SDC="${RESOLVED_SDC:-$SDC_FILE}"
  log "TOP=$TOP NETLIST=$NETLIST LIB_TGT=$LIB_TGT LIB_ROOT=$LIB_ROOT SDC_FILE=$STA_SDC RPT_DIR=$RPT_DIR SAIF_FILE=$SAIF_FILE SAIF_INST=$SAIF_INST $STA $STA_SCRIPT"
  # `set -e` would abort before the log is echoed, so a failing sta -- a
  # dynamic-loader error above all, which prints nothing on the terminal and
  # only sets exit 127 -- would leave the build with no diagnosis at all.
  sta_rc=0
  TOP=$TOP NETLIST="$NETLIST" LIB_TGT="$LIB_TGT" LIB_ROOT="$LIB_ROOT" SDC_FILE="$STA_SDC" RPT_DIR="$RPT_DIR" SAIF_FILE="$SAIF_FILE" SAIF_INST="$SAIF_INST" "$STA" "$STA_SCRIPT" > "$RPT_DIR/sta.log" 2>&1 || sta_rc=$?
  cat "$RPT_DIR/sta.log"
  if [[ $sta_rc -ne 0 ]]; then
    die "OpenSTA ('$STA') exited $sta_rc; see $RPT_DIR/sta.log"
  fi
  if [[ -n "$SAIF_FILE" ]]; then
    [[ -s "$RPT_DIR/saif_annotated.rpt" ]] || die "SAIF annotation report was not produced"
    grep -Eq '^saif[[:space:]]+[1-9][0-9]*$' "$RPT_DIR/saif_annotated.rpt" \
      || die "SAIF annotated zero pins; check SAIF_INST and the SAIF hierarchy"
  fi
  stamp "sta"
fi

echo
echo "DONE. Top: $TOP  |  RUN_SYNTH=$RUN_SYNTH  RUN_MAP=$RUN_MAP  RUN_STA=$RUN_STA"
TOTAL=$(( $(now_ms) - START_MS ))
echo "TOTAL ELAPSED: $(hhmmss $TOTAL)"

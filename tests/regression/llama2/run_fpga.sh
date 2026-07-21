#!/usr/bin/env bash
# End-to-end Llama-2 on the Alveo U55C: program the FPGA, then decode.
#
#   ./run_fpga.sh [-x <vortex.xclbin>] [-n <steps>] [-m <model.bin>] [--scalar]
#
# Stages, each gated so a failure names the actual cause:
#   1. preflight  -- XRT present, U55C present and ready
#   2. bitstream  -- locate the Vortex xclbin (NOT the platform's test ones)
#   3. program    -- load it onto the card
#   4. build      -- XRT runtime driver + llama2 host/kernel
#   5. run        -- decode end to end via VORTEX_DRIVER=xrt
#
# The Vortex bitstream is generic: it does not contain the Llama kernels, which
# are loaded at runtime from kernel.vxbin. So step 3 is done once and step 5 can
# be re-run against new kernels without re-synthesis.

set -u -o pipefail

VORTEX_HOME="${VORTEX_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
APP_DIR="$VORTEX_HOME/tests/regression/llama2"
MODEL_DIR="$VORTEX_HOME/third_party/llama2c"
XRT_DIR="${XILINX_XRT:-/opt/xilinx/xrt}"
PLATFORM="${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}"

XCLBIN=""
STEPS=32
MODEL="$MODEL_DIR/stories15M.bin"
TOKENIZER=""
EXTRA_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    -x) XCLBIN="$2"; shift 2 ;;
    -n) STEPS="$2"; shift 2 ;;
    -m) MODEL="$2"; shift 2 ;;
    --scalar) EXTRA_ARGS="$EXTRA_ARGS --scalar"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Pick the tokenizer that matches the checkpoint's vocabulary.
if [ -z "$TOKENIZER" ]; then
  case "$(basename "$MODEL")" in
    stories260K.bin) TOKENIZER="$MODEL_DIR/tok512.bin" ;;
    *)               TOKENIZER="$MODEL_DIR/tokenizer.bin" ;;
  esac
fi

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\033[31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
step "1/5  preflight"

[ -d "$XRT_DIR" ] || die "XRT not found at $XRT_DIR (set XILINX_XRT)"
# shellcheck disable=SC1091
source "$XRT_DIR/setup.sh" >/dev/null 2>&1 || die "cannot source $XRT_DIR/setup.sh"

XRT_SMI="$XRT_DIR/bin/xrt-smi"
[ -x "$XRT_SMI" ] || XRT_SMI="$XRT_DIR/bin/xbutil"
[ -x "$XRT_SMI" ] || die "neither xrt-smi nor xbutil found in $XRT_DIR/bin"

BDF=$("$XRT_SMI" examine 2>/dev/null | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]\]' | head -1 | tr -d '[]')
[ -n "$BDF" ] || die "no Xilinx accelerator found (is the card installed and the driver loaded?)"
echo "device : $BDF"
echo "shell  : $("$XRT_SMI" examine 2>/dev/null | grep -oE 'xilinx_u55c[a-z0-9_]*' | head -1)"

# --------------------------------------------------------------------------
step "2/5  locate Vortex bitstream"

if [ -z "$XCLBIN" ]; then
  # Only accept a Vortex build; the platform ships verify/bandwidth test
  # xclbins that would program fine but contain no Vortex core.
  XCLBIN=$(find "$VORTEX_HOME/hw/syn/xilinx/xrt" -name 'vortex_afu.xclbin' 2>/dev/null | head -1)
fi

if [ -z "$XCLBIN" ] || [ ! -f "$XCLBIN" ]; then
  cat >&2 <<EOF
$(printf '\033[31mFAIL:\033[0m') no Vortex xclbin found.

  The FPGA cannot be programmed with Vortex until a bitstream is built. The
  platform's own test bitstreams (verify.xclbin, bandwidth.xclbin) are NOT
  substitutes -- they contain no Vortex core and cannot run Llama.

  Build one (8-24 h; needs a config that fits the U55C -- see notes below):

    cd $VORTEX_HOME/hw/syn/xilinx/xrt
    PREFIX=llama NUM_CORES=1 TARGET=hw PLATFORM=$PLATFORM \\
      CONFIGS="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=8 \\
               -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE \\
               -DVX_CFG_TCU_WGMMA_ENABLE" \\
      make > build.log 2>&1 &

  Then re-run with:  $0 -x <path-to-vortex_afu.xclbin>
EOF
  exit 1
fi
echo "xclbin : $XCLBIN"

# --------------------------------------------------------------------------
step "3/5  program FPGA"

"$XRT_SMI" program -d "$BDF" -u "$XCLBIN" || die "programming failed"
echo "programmed OK"

# --------------------------------------------------------------------------
step "4/5  build XRT driver + application"

# The app's CONFIGS must match the config baked into the xclbin exactly --
# NUM_THREADS in particular sets the WGMMA tile geometry the host launch
# bounds are derived from. Override APP_CONFIGS if you built a different core.
APP_CONFIGS="${APP_CONFIGS:--DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DPERF_ENABLE}"
echo "app configs: $APP_CONFIGS"
# TARGET=hw builds the real-hardware XRT driver (sw/runtime/xrt/vortex.cpp against
# the XRT libs). The Makefile default is xrtsim, which verilates a sim model and
# is not what talks to the card.
make -C "$VORTEX_HOME/sw/runtime/xrt" TARGET=hw CONFIGS="$APP_CONFIGS" >/dev/null 2>&1 || die "XRT runtime driver build failed"
make -C "$APP_DIR" CONFIGS="$APP_CONFIGS" >/dev/null 2>&1 || die "llama2 build failed"
echo "built OK"

# --------------------------------------------------------------------------
step "5/5  run Llama end to end on hardware"

[ -f "$MODEL" ]     || die "model not found: $MODEL"
[ -f "$TOKENIZER" ] || die "tokenizer not found: $TOKENIZER"
echo "model     : $MODEL"
echo "tokenizer : $TOKENIZER"
echo "steps     : $STEPS"
echo

cd "$APP_DIR" || die "cannot enter $APP_DIR"
LD_LIBRARY_PATH="$VORTEX_HOME/sw/runtime:${LD_LIBRARY_PATH:-}" \
VORTEX_DRIVER=xrt \
XRT_XCLBIN_PATH="$XCLBIN" \
  ./llama2 "$MODEL" -z "$TOKENIZER" -n "$STEPS" $EXTRA_ARGS
rc=$?

echo
[ $rc -eq 0 ] && printf '\033[32mDONE\033[0m  end-to-end Llama ran on %s\n' "$BDF" \
              || printf '\033[31mFAIL\033[0m  run exited %d\n' "$rc"
exit $rc

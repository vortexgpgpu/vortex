# Environment for building and running Vortex through FireSim. Source it from
# the build directory, after setting up the card's Xilinx tools exactly as the
# native flow for that card needs them (Vitis and XRT for an XRT card; Vivado
# and VRT for an aved card):
#
#   source ci/firesim_env.sh
#
# It adds only what FireSim needs on top of those:
#   - FireSim's conda environment, through its env.sh (conda must be on PATH);
#   - the host compiler and linker, which VRT and the aved simulator build
#     need: conda's cannot link against the host glibc those libraries use;
#   - Vivado's simulator libraries, which an aved TARGET=sim run loads.
#
#   FIRESIM_PATH     the FireSim fork checkout (default: config.mk's)
#   FIRESIM_ENV_SH   FireSim's env.sh, if its conda environment lives in another
#                    checkout (default: $FIRESIM_PATH/env.sh)

_fs_fail() { echo "firesim_env: $*" >&2; return 1; }

_fs_main() {
    if [ -z "${FIRESIM_PATH:-}" ] && [ -f config.mk ]; then
        FIRESIM_PATH=$(make -s -f config.mk -f - <<<'print-fs: ; @echo $(FIRESIM_PATH)' print-fs 2>/dev/null)
    fi
    [ -n "${FIRESIM_PATH:-}" ] || { _fs_fail "set FIRESIM_PATH, or source this from a configured build directory"; return 1; }
    [ -d "$FIRESIM_PATH/sim" ] || { _fs_fail "no FireSim checkout at FIRESIM_PATH=$FIRESIM_PATH"; return 1; }
    export FIRESIM_PATH

    command -v conda >/dev/null 2>&1 || { _fs_fail "conda is not on PATH (FireSim's env.sh needs it)"; return 1; }
    local env_sh=${FIRESIM_ENV_SH:-$FIRESIM_PATH/env.sh}
    [ -f "$env_sh" ] || { _fs_fail "FireSim's env.sh not found at $env_sh (set FIRESIM_ENV_SH)"; return 1; }
    # env.sh reads variables it has not set, so it cannot run under set -u.
    local had_u=0; case $- in *u*) had_u=1; set +u ;; esac
    source "$env_sh" >/dev/null 2>&1
    [ $had_u = 1 ] && set -u
    # env.sh may repoint FIRESIM_PATH at its own checkout; keep the one asked for.
    export FIRESIM_PATH

    # The host compiler, and the host linker ahead of conda's under the plain
    # name `ld` that CMake-driven builds invoke.
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    local hostld=${XDG_CACHE_HOME:-$HOME/.cache}/vortex-firesim/hostld
    mkdir -p "$hostld" && ln -sf /usr/bin/ld "$hostld/ld"
    case ":$PATH:" in *":$hostld:"*) ;; *) export PATH="$hostld:$PATH" ;; esac

    if [ -n "${XILINX_VIVADO:-}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$XILINX_VIVADO/lib/lnx64.o:"*) ;;
            *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$XILINX_VIVADO/lib/lnx64.o" ;;
        esac
    else
        echo "firesim_env: XILINX_VIVADO is not set; set up the card's Xilinx tools before building" >&2
    fi

    # A stale VRT ahead of the packaged one on the library path runs every V80
    # image at the shell's boot clock instead of the frequency it was built for.
    case ":${LD_LIBRARY_PATH:-}:" in
        *":/opt/xilinx/slash/lib:"*)
            echo "firesim_env: warning: /opt/xilinx/slash/lib is on LD_LIBRARY_PATH; make sure it is the same VRT as the installed vrtd" >&2 ;;
    esac

    echo "firesim_env: FIRESIM_PATH=$FIRESIM_PATH"
}

_fs_main
unset -f _fs_main _fs_fail

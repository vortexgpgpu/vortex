# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Build fragment for device fragment-kernels that use the SIMT software
# output-merger sw/common/gfx_sw.h::om_fragment (the gfx_v2 §6.5 SW back end).
#
# om_fragment's full depth + stencil + blend + logic-op merge inlines a large
# divergent CFG into the fragment kernel — past the Vortex divergence pass's
# default 100-BB guard. If the guard trips, the pass silently skips
# StructurizeCFG + split/join and the kernel miscompiles (uniform OM-state reads
# left as unselectable markers, divergent control flow unmasked). Raising the
# guard fixes it; this fragment encapsulates that flag so no call site can forget
# it, and defines GFX_SW_DIVERGENCE_OK — without which gfx_sw.h #errors.
#
# Usage (in a test/kernel Makefile):
#   include $(VORTEX_HOME)/sw/gfx/libgfx_sw.mk
#   VX_CFLAGS += $(LIBGFX_SW_VX_CFLAGS)
#
# The bound is sized well above any single fragment kernel's block count; raise
# it if a kernel ever exceeds it (the #error / a miscompile is the signal).
LIBGFX_SW_VX_CFLAGS := -mllvm -vortex-divergence-max-bbs=512 -DGFX_SW_DIVERGENCE_OK

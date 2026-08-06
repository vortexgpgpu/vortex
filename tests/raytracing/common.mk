# tests/raytracing/common.mk — forwards to the canonical
# tests/regression/common.mk so the RTU tests share the same
# build/run rules. Lets us reorganise tests/raytracing/ later
# without copy-pasting common.mk per group.
include $(realpath $(dir $(lastword $(MAKEFILE_LIST))))/../regression/common.mk

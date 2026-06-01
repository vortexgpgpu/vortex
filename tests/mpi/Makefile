ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# MPI multi-process tests. Driven by ci/regression.sh's mpi() with
# `MPI=1 NP=4` per subdir (mpic++ + mpirun). MPI/NP pass through to the
# subdir Makefiles so `make -C tests/mpi run-simx MPI=1 NP=4` works too.
# Default MPI=0 (single process) matches the per-test Makefile default.
TESTS := \
    mpi_dotproduct mpi_put_dotproduct mpi_diverge mpi_vecadd \
    mpi_sgemm mpi_blocked_sgemm mpi_conv3 mpi_neighbor_a2a_conv3

MPI ?= 0
NP  ?= 1

BACKENDS := simx rtlsim opae xrt

.PHONY: all clean $(addprefix run-,$(BACKENDS))

all:   $(addprefix build-,$(TESTS))
clean: $(addprefix clean-,$(TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(TESTS))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

.PHONY: build-% clean-%

build-%:
	$(MAKE) -C $* all MPI=$(MPI) NP=$(NP)

clean-%:
	$(MAKE) -C $* clean

# run-<backend>-<test>: split "<backend>-<test>" on the first dash.
# Subdir names must not contain '-' (none currently do).
run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*)) MPI=$(MPI) NP=$(NP)

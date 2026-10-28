# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# tess -- tessellation on the device: control and evaluation shaders around
# the SIMT tessellator, every domain, spacing and winding, point mode, levels
# known on the host or only on the device, patch lists cut by a restart index,
# compared pixel for pixel with the host's own tessellation of each patch drawn
# as a list.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case but levels_max (OPTS="-c <case>" for one)
#
# levels_max, 956 patches at level 64 with the widest evaluation records, is
# run on its own, on the CTS device shape -- four cores, functional SimX:
# make run-simx OPTS="-c levels_max" EXPECT_LINES=.
#
# STRICT stays at the suite default of 1: llvmpipe tessellates correctly, so a
# draw that fell back would pass without testing the device. WIDE because a
# patch's control points are one CTA, and a device offers tessellation only
# with patches of 32 control points, more than a 4x4 core's CTA holds. The
# run's tessellation attribution pins how many draws took the path with levels
# known on the host. handoff, which hands a capture to llvmpipe and back, runs
# on its own with STRICT 0: make run-simx STRICT=0 OPTS="-c handoff"
# EXPECT_CPU=transform-feedback=1 EXPECT_LINES=.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tess

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

SHADERS := $(SRC_DIR)/tess.vert $(SRC_DIR)/tess.tesc $(SRC_DIR)/pass.tesc \
           $(SRC_DIR)/pass.tese $(SRC_DIR)/tess.frag $(SRC_DIR)/ref.vert \
           $(SRC_DIR)/gs_tri.geom $(SRC_DIR)/gs_line.geom $(SRC_DIR)/gs_point.geom \
           $(SRC_DIR)/gs_amp.geom $(SRC_DIR)/max.tese $(SRC_DIR)/barrier.tesc \
           $(SRC_DIR)/outer.tesc $(SRC_DIR)/gs_xfb.geom

# The reference tessellates on the host with the device's own tessellator.
CC := g++
CFLAGS += -std=c++17 -I$(VORTEX_HOME)/sw/common

WIDE := 1
EXPECT_LINES ?= vortexpipe-tess:_static=$(TESS_STATIC)_dynamic=$(TESS_DYNAMIC)
TESS_STATIC := 23
TESS_DYNAMIC := 38

ifneq ($(filter levels_max,$(OPTS)),)
CONFIGS += -DVX_CFG_NUM_CORES=4 -DVX_CFG_L2_ENABLE -DSIMX_FUNCTIONAL
endif

include ../common.mk

# One evaluation shader per domain, spacing, winding and point mode, from
# tess.tese: <domain>_<spacing>_<winding>, iso_<spacing>, <domain>_points
# (fractional odd), quad_id (coloured by gl_PrimitiveID), *_xfb (capturing
# the colour), and *_wide (outputs no later stage reads).
comma := ,
TESE_VARIANTS := $(foreach d,tri quad,$(foreach s,equal odd even,$(foreach w,cw ccw,$(d)_$(s)_$(w)))) \
                 iso_equal iso_odd iso_even tri_points quad_points iso_points quad_id \
                 quad_equal_ccw_xfb iso_even_xfb quad_odd_ccw_wide quad_odd_ccw_view
TESE_SPVS := $(addsuffix .tese.spv,$(TESE_VARIANTS))
tese_dom = $(if $(filter tri%,$1),triangles,$(if $(filter quad%,$1),quads,isolines))
tese_spc = $(if $(findstring _equal,$1),equal_spacing,$(if $(findstring _even,$1),fractional_even_spacing,fractional_odd_spacing))
tese_wnd = $(if $(findstring _cw,$1),cw,ccw)
tese_pts = $(if $(findstring _points,$1),$(comma)point_mode,)

$(TESE_SPVS): %.tese.spv: $(SRC_DIR)/tess.tese
	$(GLSLC) -fshader-stage=tese -DDOMAIN=$(call tese_dom,$*) -DSPACING=$(call tese_spc,$*) \
	   -DWINDING=$(call tese_wnd,$*) "-DPOINTS=$(call tese_pts,$*)" \
	   -DTRI=$(if $(filter tri%,$*),true,false) $(if $(filter quad_id,$*),-DID) \
	   $(if $(findstring _xfb,$*),-DXFB) $(if $(findstring _wide,$*),-DWIDE) \
	   $(if $(findstring _view,$*),-DVIEW) $< -o $@

# The control shader with 16 and 32 control points, and barrier.tesc with its
# rounds in a loop, from its SPIR-V (see barrier_nested.spvasm).
TESC_SPVS := tess_m16.tesc.spv tess_m32.tesc.spv barrier_nested.tesc.spv
tess_m%.tesc.spv: $(SRC_DIR)/tess.tesc
	$(GLSLC) -fshader-stage=tesc -DNV=$* $< -o $@
barrier_nested.tesc.spv: $(SRC_DIR)/barrier_nested.spvasm
	$(GLSLC) $< -o $@

SPVS += $(TESE_SPVS) $(TESC_SPVS)
all: $(TESE_SPVS) $(TESC_SPVS)

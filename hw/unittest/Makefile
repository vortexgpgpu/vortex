all:
	$(MAKE) -C generic_queue
	$(MAKE) -C mem_scheduler
	$(MAKE) -C cache
	$(MAKE) -C core
	$(MAKE) -C issue
	$(MAKE) -C local_mem
	$(MAKE) -C mem_unit
	$(MAKE) -C tcu_fedp
	$(MAKE) -C cta_dispatcher
	$(MAKE) -C kmu
	$(MAKE) -C dxa_core
	$(MAKE) -C tcu_unit
	$(MAKE) -C cp_arbiter
	$(MAKE) -C cp_engine
	$(MAKE) -C cp_launch
	$(MAKE) -C cp_dcr_proxy
	$(MAKE) -C cp_unpack
	$(MAKE) -C cp_axil_regfile
	$(MAKE) -C afu_axil_demux
	$(MAKE) -C afu_reset_seq
	$(MAKE) -C mm_axi_arb
	$(MAKE) -C cp_axi_path
	$(MAKE) -C cp_dma
	$(MAKE) -C cp_llc_port
	$(MAKE) -C stream_arb
	$(MAKE) -C cp_core
	$(MAKE) -C fma_unit
	$(MAKE) -C fdiv_unit
	$(MAKE) -C fsqrt_unit
	$(MAKE) -C fcvt_unit
	$(MAKE) -C fdivsqrt_unit
	$(MAKE) -C tex_lerp
	$(MAKE) -C mufu_unit

run:
	$(MAKE) -C generic_queue run
	$(MAKE) -C mem_scheduler run
	$(MAKE) -C cache run
	$(MAKE) -C core run
	$(MAKE) -C issue run
	$(MAKE) -C local_mem run
	$(MAKE) -C mem_unit run
	$(MAKE) -C tcu_fedp run
	$(MAKE) -C cta_dispatcher run
	$(MAKE) -C kmu run
	$(MAKE) -C dxa_core run
	$(MAKE) -C tcu_unit run
	$(MAKE) -C cp_arbiter run
	$(MAKE) -C cp_engine run
	$(MAKE) -C cp_launch run
	$(MAKE) -C cp_dcr_proxy run
	$(MAKE) -C cp_unpack run
	$(MAKE) -C cp_axil_regfile run
	$(MAKE) -C afu_axil_demux run
	$(MAKE) -C afu_reset_seq run
	$(MAKE) -C mm_axi_arb run
	$(MAKE) -C cp_axi_path run
	$(MAKE) -C cp_dma run
	$(MAKE) -C cp_llc_port run
	$(MAKE) -C stream_arb run
	$(MAKE) -C cp_core run
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run
	$(MAKE) -C tex_lerp run
	$(MAKE) -C mufu_unit run

# FPU arithmetic units only — executed (not just built) so CI gates correctness.
run-fpu:
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run

# MUFU datapath, bit-exact against the sw/common golden model.
run-mufu:
	$(MAKE) -C mufu_unit run

# TCU FEDP datapath — executed per format so a tcu/tfr change is gated by CI.
# Each format must enable its own RTL datapath (FP16-only is the default). Covers
# the native TFR and the DPI reference. Float formats compare against the windowed
# FEDP reference (USE_FEDP) with the datapath window --W=25 and the per-format
# NUM_REGS; DPI compares discrete mul/add (--no-fused). Integer formats use the
# exact reference (no window). Switches mirror ci/regression.sh's tensor() suite.
TCU_BASE := -DSIMULATION -DSV_DPI -DVX_CFG_EXT_TCU_ENABLE
# USE_DSP=1 maps the native-TFR FEDP multipliers onto DSP48 (vs the default
# LUT/Wallace path). Forwarded to tcu_fedp; ignored by the non-TFR datapaths.
USE_DSP ?= 0
# $(1)=NUM_REGS $(2)=extra CONFIGS $(3)=run OPTS
define tcu_case
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp NUM_REGS=$(1) USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) $(2)"
	$(MAKE) -C tcu_fedp run NUM_REGS=$(1) USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) $(2)" OPTS="$(3)"
endef
# integer pair on one build (int8/uint8 share INT8; int4/uint4 share INT4)
# $(1)=type $(2)=enable $(3)=fmtA $(4)=fmtB
define tcu_int_pair
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE"
	$(MAKE) -C tcu_fedp run USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(3)"
	$(MAKE) -C tcu_fedp run USE_DSP=$(USE_DSP) CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(4)"
endef
# Native-TFR float + integer formats; USE_DSP comes from the caller.
# The testbench latency derives from the configured VX_CFG_TCU_LATENCY; the
# last case pins the 2-stage multiply used for FPGA synthesis
# (VX_CFG_TCU_LATENCY=5).
define tcu_tfr_suite
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=2 --W=25)
	$(call tcu_case,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=3 --W=25)
	$(call tcu_case,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --W=25)
	$(call tcu_case,2,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_LATENCY=5,--fmt=2 --W=25)
	$(call tcu_int_pair,TFR,INT8,17,18)
	$(call tcu_int_pair,TFR,INT4,19,20)
endef
run-tcu:
	# native TFR — LUT/Wallace multipliers (USE_DSP=0)
	$(call tcu_tfr_suite)
	# DPI reference — float (discrete mul/add, --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI,--fmt=2 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI,--fmt=3 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --no-fused)
	$(call tcu_case,1,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --no-fused)
	# DPI reference — integer
	$(call tcu_int_pair,DPI,INT8,17,18)
	$(call tcu_int_pair,DPI,INT4,19,20)

# VX_tcu_unit end to end (dispatch, uops, operand routing, commit), bit for bit
# (--ulp=0) against the reference of each datapath: the native TFR one against
# the TfrFedp reference model, the DPI one against its discrete fp32 mul/add.
# One build per tile shape and datapath: 4 threads (2x2) with tf32/fp16/bf16,
# 8 threads (4x2) fp16 on DPI.
TCU_UNIT_TFR := $(TCU_BASE) -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_TCU_TF32_ENABLE
TCU_UNIT_DPI := $(TCU_BASE) -DVX_CFG_TCU_TYPE_DPI
run-tcu-unit:
	$(MAKE) -C tcu_unit clean
	$(MAKE) -C tcu_unit CONFIGS="$(TCU_UNIT_TFR)"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_TFR)" OPTS="--fmt=1 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_TFR)" OPTS="--fmt=2 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_TFR)" OPTS="--fmt=3 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit clean
	$(MAKE) -C tcu_unit CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_TCU_TF32_ENABLE"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_TCU_TF32_ENABLE" OPTS="--fmt=1 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_TCU_TF32_ENABLE" OPTS="--fmt=2 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_TCU_TF32_ENABLE" OPTS="--fmt=3 --ulp=0 --no-trace"
	$(MAKE) -C tcu_unit clean
	$(MAKE) -C tcu_unit CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_NUM_THREADS=8"
	$(MAKE) -C tcu_unit run CONFIGS="$(TCU_UNIT_DPI) -DVX_CFG_NUM_THREADS=8" OPTS="--fmt=2 --ulp=0 --no-trace"
	$(call tcu_mx_suite,2,1)
	$(call tcu_mx_suite,4,2)

# The native datapath's fp8 and block-scaled formats against the TfrFedp
# reference model, bit for bit: fp8/bf8, MXFP8/MXBF8 (E8M0 scales), MXFP4 (E8M0)
# and NVFP4 (E4M3). VX_tcu_unit takes its block scales from the metadata path,
# so these run on the FEDP. Two shapes: 2 registers on one scale slot, 4 on two.
# Directed legs pin zero products in an fp8 lane, MXFP4 scales far from one
# (2^-34), MXFP8/MXBF8 scales so small the lane underflows, and NaN scales.
TCU_MX := $(TCU_BASE) -DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP8_ENABLE \
          -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE
# format:sf_a:sf_b
TCU_MX_DIRECTED := 10:110:110 8:0:3 9:4:2 8:0xff:127 11:0x7f:0x38
tcu_mx_opts = --fmt=$(word 1,$(subst :, ,$(1))) --sfa=$(word 2,$(subst :, ,$(1))) --sfb=$(word 3,$(subst :, ,$(1)))
# $(1)=NUM_REGS $(2)=SF
define tcu_mx_suite
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp NUM_REGS=$(1) SF=$(2) CONFIGS="$(TCU_MX)"
	$(foreach f,4 5 8 9 10 11,$(MAKE) -C tcu_fedp run NUM_REGS=$(1) SF=$(2) CONFIGS="$(TCU_MX)" OPTS="--fmt=$(f) --W=25 --ulp=0 --no-trace" &&) true
	$(foreach f,4 8 9,$(MAKE) -C tcu_fedp run NUM_REGS=$(1) SF=$(2) CONFIGS="$(TCU_MX)" OPTS="--fmt=$(f) --W=25 --ulp=0 --features=zeros --tests=20000 --no-trace" &&) true
	$(foreach o,$(TCU_MX_DIRECTED),$(MAKE) -C tcu_fedp run NUM_REGS=$(1) SF=$(2) CONFIGS="$(TCU_MX)" OPTS="$(call tcu_mx_opts,$(o)) --W=25 --ulp=0 --tests=20000 --no-trace" &&) true
endef

# Same native-TFR formats as run-tcu, but with the multipliers mapped onto DSP48
# (USE_DSP=1). USE_DSP is bit-identical to the LUT path by contract, so the same
# references apply — a mismatch here is a DSP-packing bug.
run-tcu-dsp: USE_DSP := 1
run-tcu-dsp:
	$(call tcu_tfr_suite)

clean:
	$(MAKE) -C generic_queue clean
	$(MAKE) -C mem_scheduler clean
	$(MAKE) -C cache clean
	$(MAKE) -C core clean
	$(MAKE) -C issue clean
	$(MAKE) -C local_mem clean
	$(MAKE) -C mem_unit clean
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C cta_dispatcher clean
	$(MAKE) -C kmu clean
	$(MAKE) -C dxa_core clean
	$(MAKE) -C tcu_unit clean
	$(MAKE) -C cp_arbiter clean
	$(MAKE) -C cp_engine clean
	$(MAKE) -C cp_launch clean
	$(MAKE) -C cp_dcr_proxy clean
	$(MAKE) -C cp_unpack clean
	$(MAKE) -C cp_axil_regfile clean
	$(MAKE) -C cp_axi_path clean
	$(MAKE) -C cp_dma clean
	$(MAKE) -C cp_llc_port clean
	$(MAKE) -C stream_arb clean
	$(MAKE) -C mm_axi_arb clean
	$(MAKE) -C cp_core clean
	$(MAKE) -C fma_unit clean
	$(MAKE) -C fdiv_unit clean
	$(MAKE) -C fsqrt_unit clean
	$(MAKE) -C fcvt_unit clean
	$(MAKE) -C fdivsqrt_unit clean
	$(MAKE) -C tex_lerp clean
	$(MAKE) -C mufu_unit clean

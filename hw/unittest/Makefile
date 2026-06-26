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
	$(MAKE) -C cp_axi_path
	$(MAKE) -C cp_dma
	$(MAKE) -C cp_core
	$(MAKE) -C fma_unit
	$(MAKE) -C fdiv_unit
	$(MAKE) -C fsqrt_unit
	$(MAKE) -C fcvt_unit
	$(MAKE) -C fdivsqrt_unit

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
	$(MAKE) -C cp_axi_path run
	$(MAKE) -C cp_dma run
	$(MAKE) -C cp_core run
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run

# FPU arithmetic units only — executed (not just built) so CI gates correctness.
run-fpu:
	$(MAKE) -C fma_unit run
	$(MAKE) -C fdiv_unit run
	$(MAKE) -C fsqrt_unit run
	$(MAKE) -C fcvt_unit run
	$(MAKE) -C fdivsqrt_unit run

# TCU FEDP datapath — executed per format so a tcu/tfr change is gated by CI.
# Each format must enable its own RTL datapath (FP16-only is the default). Covers
# the native TFR and the DPI reference. Float formats compare against the windowed
# FEDP reference (USE_FEDP) with the datapath window --W=25 and the per-format
# NUM_REGS; DPI compares discrete mul/add (--no-fused). Integer formats use the
# exact reference (no window). Switches mirror ci/regression.sh's tensor() suite.
TCU_BASE := -DSIMULATION -DSV_DPI -DVX_CFG_EXT_TCU_ENABLE
# $(1)=NUM_REGS $(2)=LATENCY $(3)=extra CONFIGS $(4)=run OPTS
define tcu_case
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp NUM_REGS=$(1) LATENCY=$(2) CONFIGS="$(TCU_BASE) $(3)"
	$(MAKE) -C tcu_fedp run NUM_REGS=$(1) LATENCY=$(2) CONFIGS="$(TCU_BASE) $(3)" OPTS="$(4)"
endef
# integer pair on one build (int8/uint8 share INT8; int4/uint4 share INT4)
# $(1)=type $(2)=enable $(3)=fmtA $(4)=fmtB
define tcu_int_pair
	$(MAKE) -C tcu_fedp clean
	$(MAKE) -C tcu_fedp CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE"
	$(MAKE) -C tcu_fedp run CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(3)"
	$(MAKE) -C tcu_fedp run CONFIGS="$(TCU_BASE) -DVX_CFG_TCU_TYPE_$(1) -DVX_CFG_TCU_$(2)_ENABLE" OPTS="--fmt=$(4)"
endef
run-tcu:
	# native TFR — float (USE_FEDP windowed reference, --W=25)
	$(call tcu_case,2,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --W=25)
	$(call tcu_case,2,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=2 --W=25)
	$(call tcu_case,4,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP,--fmt=3 --W=25)
	$(call tcu_case,4,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --W=25)
	$(call tcu_case,2,4,-DVX_CFG_TCU_TYPE_TFR -DUSE_FEDP -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --W=25)
	# native TFR — integer (exact reference)
	$(call tcu_int_pair,TFR,INT8,17,18)
	$(call tcu_int_pair,TFR,INT4,19,20)
	# DPI reference — float (discrete mul/add, --no-fused)
	$(call tcu_case,1,4,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_TF32_ENABLE,--fmt=1 --no-fused)
	$(call tcu_case,1,4,-DVX_CFG_TCU_TYPE_DPI,--fmt=2 --no-fused)
	$(call tcu_case,1,4,-DVX_CFG_TCU_TYPE_DPI,--fmt=3 --no-fused)
	$(call tcu_case,1,4,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=4 --no-fused)
	$(call tcu_case,1,4,-DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_FP8_ENABLE,--fmt=5 --no-fused)
	# DPI reference — integer
	$(call tcu_int_pair,DPI,INT8,17,18)
	$(call tcu_int_pair,DPI,INT4,19,20)

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
	$(MAKE) -C cp_core clean
	$(MAKE) -C fma_unit clean
	$(MAKE) -C fdiv_unit clean
	$(MAKE) -C fsqrt_unit clean
	$(MAKE) -C fcvt_unit clean
	$(MAKE) -C fdivsqrt_unit clean

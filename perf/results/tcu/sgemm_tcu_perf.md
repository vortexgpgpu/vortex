# TCU Perf Snapshot

Last updated: 01-07-2026

## SimX Commands

```bash
# sgemm_tcu
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE -DITYPE=fp16 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE -DITYPE=fp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE -DITYPE=int8 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE -DITYPE=int4 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_sp
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=fp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=int8 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=int4 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_mx
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxfp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DITYPE=mxfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DITYPE=nvfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxint8 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_sp_mx
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxfp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DITYPE=mxfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DTCU_MX_TLS -DITYPE=nvfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxint8 -DOTYPE=int32" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1
```

## RTLsim Commands

Swap out -DVX_CFG_TCU_TYPE_DPI backend for -DVX_CFG_TCU_TYPE_TFR or -DVX_CFG_TCU_TYPE_TET as required

```bash
# sgemm_tcu
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DITYPE=fp16 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DITYPE=fp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DITYPE=int8 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DITYPE=int4 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_sp
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=fp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=int8 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=int4 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_mx
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxfp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DITYPE=mxfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DITYPE=nvfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxint8 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_mx --args="-m64 -n64 -k64" --perf=1

# sgemm_tcu_sp_mx
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxfp8 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_MXFP4_ENABLE -DITYPE=mxfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_FP4_ENABLE -DVX_CFG_TCU_NVFP4_ENABLE -DTCU_MX_TLS -DITYPE=nvfp4 -DOTYPE=fp32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1

CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_NUM_WARPS=8 -DVX_CFG_EXT_TCU_ENABLE  -DVX_CFG_TCU_TYPE_DPI -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DITYPE=mxint8 -DOTYPE=int32" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_sp_mx --args="-m64 -n64 -k64" --perf=1
```

## FPGA Synthesis Configs

Target Clock Frequency: 300 MHz
Post-Implementation results reported

TFR runs:
```bash
# tcu
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16"

# tcu_sp
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16 -DVX_CFG_TCU_SPARSE_ENABLE"

# tcu_mx
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16 -DVX_CFG_TCU_MX_ENABLE"

# tcu_sp_mx
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16 -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE"

```

TET runs:
```bash
# tcu
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TET -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16"

# tcu_mx
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE_TET -DVX_CFG_NUM_THREADS=16 -DVX_CFG_TCU_NUM_WARPS=16 -DVX_CFG_TCU_MX_ENABLE"
```
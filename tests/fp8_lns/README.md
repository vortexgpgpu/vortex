# FP8 + LNS Software Experiment

First-stage numerical test harness for comparing:

1. FP32 baseline
2. FP8 operands with FP32 accumulation
3. FP8 storage -> quantized LNS multiply -> FP32 accumulation

This is intentionally a **software emulation** experiment before changing Vortex RTL.

## Setup

```bash
make setup
```

## Run

```bash
make e4m3
make e5m2
make sweep
make test
```

You can also run directly:

```bash
.venv/bin/python fp8_lns_experiment.py \
  --format e4m3 \
  --lns-int 3 \
  --lns-frac 4 \
  --matrix 64
```

## What is being tested?

For the hybrid path, operands are first quantized to FP8. Their magnitudes are converted to quantized log2 values, multiplication is performed as addition in the log domain, the product is converted back to linear space, and accumulation stays FP32.

Conceptually:

```text
FP8 A ----> quantized log2(A) --\
                                  + --> LNS product --> linear product --\
FP8 B ----> quantized log2(B) --/                                      +--> FP32 accumulator
```

## First research question

Does the error introduced by the LNS multiplier remain acceptable relative to plain FP8?

After numerical behavior is characterized, the next step is to port the useful configurations into a Vortex kernel/simulator experiment and separately estimate hardware cost.

## Important limitation

The FP8 functions here are arithmetic quantizers, not bit-exact implementations of every special-case rule in NVIDIA/Intel FP8 specifications. That is deliberate for the first-stage comparison. If the experiment looks promising, the next revision should implement exact E4M3/E5M2 encodings and match the intended Vortex datapath semantics.

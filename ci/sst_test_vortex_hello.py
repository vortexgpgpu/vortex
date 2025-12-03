import sst

gpu = sst.Component("gpu0", "vortex.VortexGPGPU")
gpu.addParams({
    "clock": "1GHz",
    "program": "../build/tests/kernel/hello/hello.bin"
})

#include "vec_unit.h"
#include "vec_ops.h"
#include "core.h"

using namespace vortex;

// Simulate clock cycles depending on instruction type and element width and #lanes
// VSET = 1 cycle
// Vector instructions take the same amount of time as ALU instructions.
// In general there should be less overall instructions (hence the SIMD vector speedup).
// But, each vector instruction is bigger, and # of lanes greatly effects execution speed.

// Whenever we change VL using imm/VSET, we need to keep track of the new VL and SEW.
// By default, VL is set to MAXVL.
// After determining VL, we use VL and #lanes in order to determine overall cycle time.
// For example, for a vector add with VL=4 and #lanes=2, we will probably take 2 cycles,
// since we can only operate on two elements of the vector each cycle (limited by #lanes).
// SEW (element width) likely affects the cycle time, we can probably observe
// ALU operation cycle time in relation to element width to determine this though.

// The RTL implementation has an unroll and accumulate stage.
// The unroll stage sends vector elements to the appropriate functional unit up to VL,
// limited by the # lanes available.
// The accumulate stage deals with combining the results from the functional units,
// into the destination vector register.
// Which exact pipeline stage does the VPU unroll the vector (decode or execute)?
// Which exact pipeline stage does the VPU accumulate results?

// How do vector loads and stores interact with the cache?
// How about loading and storing scalars in vector registers?
// How does striding affect loads and stores?

class VecUnit::Impl {
public:
  Impl(VecUnit* simobject, const Arch& arch, Core* core)
      : simobject_(simobject)
      , core_(core)
      , vpu_states_(arch.num_warps(), arch.num_threads())
      , num_lanes_(arch.num_warps())
      , pending_reqs_(arch.num_warps())
  {
    this->reset();
  }

  ~Impl() {}

  void reset() {
    pending_reqs_.clear();
    perf_stats_ = PerfStats();
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
          return;

      auto trace = input.front();

      int delay = 0;
      switch (trace->vpu_type) {
      case VpuType::VSET:
        break;
      case VpuType::ARITH:
      case VpuType::ARITH_R:
        delay = 1;
        break;
      case VpuType::IMUL:
        delay = LATENCY_IMUL;
        break;
      case VpuType::IDIV:
        delay = XLEN;
        break;
      case VpuType::FNCP:
      case VpuType::FNCP_R:
        delay = 2;
        break;
      case VpuType::FMA:
      case VpuType::FMA_R:
        delay = LATENCY_FMA;
        break;
      case VpuType::FDIV:
        delay = LATENCY_FDIV;
        break;
      case VpuType::FSQRT:
        delay = LATENCY_FSQRT;
        break;
      case VpuType::FCVT:
        delay = LATENCY_FCVT;
        break;
      default:
        std::abort();
      }

      simobject_->Outputs.at(iw).push(trace, 2 + delay);

      DT(3, simobject_->name() << ": op=" << trace->vpu_type << ", " << *trace);

      input.pop();
    }
  }

  void load(const Instr &instr,
            uint32_t wid,
            uint32_t tid,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            MemTraceData* trace_data) {
    auto& states = vpu_states_.at(wid);
    uint32_t vmask = instr.getVmask();
    uint32_t vd = instr.getDestReg().idx;
    uint32_t mop = instr.getVmop();
    uint32_t vsewb = 1 << states.vtype.vsew;
    auto& vreg_file = states.vreg_file.at(tid);
    uint64_t base_addr = rs1_data.at(tid).i;
    base_addr &= 0xFFFFFFFC; // TODO: riscv-tests fix

    // udpate trace data
    trace_data->vl = states.vl;
    trace_data->vnf = instr.getVnf() + 1;

    switch (mop) {
    case 0b00: { // unit-stride
      auto lumop = instr.getVumop();
      switch (lumop) {
      case 0b00000: // vle8.v, vle16.v, vle32.v, vle64.v
                    // vlseg2e8.v, vlseg2e16.v, vlseg2e32.v, vlseg2e64.v
                    // vlseg3e8.v, vlseg3e16.v, vlseg3e32.v, vlseg3e64.v
                    // vlseg4e8.v, vlseg4e16.v, vlseg4e32.v, vlseg4e64.v
                    // vlseg5e8.v, vlseg5e16.v, vlseg5e32.v, vlseg5e64.v
                    // vlseg6e8.v, vlseg6e16.v, vlseg6e32.v, vlseg6e64.v
                    // vlseg7e8.v, vlseg7e16.v, vlseg7e32.v, vlseg7e64.v
                    // vlseg8e8.v, vlseg8e16.v, vlseg8e32.v, vlseg8e64.v
      case 0b10000:{// vle8ff.v, vle16ff.v, vle32ff.v, vle64ff.v - we do not support exceptions -> treat like regular unit stride
                    // vlseg2e8ff.v, vlseg2e16ff.v, vlseg2e32ff.v, vlseg2e64ff.v
                    // vlseg3e8ff.v, vlseg3e16ff.v, vlseg3e32ff.v, vlseg3e64ff.v
                    // vlseg4e8ff.v, vlseg4e16ff.v, vlseg4e32ff.v, vlseg4e64ff.v
                    // vlseg5e8ff.v, vlseg5e16ff.v, vlseg5e32ff.v, vlseg5e64ff.v
                    // vlseg6e8ff.v, vlseg6e16ff.v, vlseg6e32ff.v, vlseg6e64ff.v
                    // vlseg7e8ff.v, vlseg7e16ff.v, vlseg7e32ff.v, vlseg7e64ff.v
                    // vlseg8e8ff.v, vlseg8e16ff.v, vlseg8e32ff.v, vlseg8e64ff.v
        uint32_t nfields = instr.getVnf() + 1;
        uint32_t emul = (states.vtype.vlmul >> 2) ? 1 : (1 << (states.vtype.vlmul & 0b11));
        assert(nfields * emul <= 8);

        for (uint32_t i = 0; i < states.vl; i++) {
          if (isMasked(vreg_file, 0, i, vmask))
            continue;
          for (uint32_t f = 0; f < nfields; f++) {
            uint64_t mem_addr = base_addr + (i * nfields + f) * vsewb;
            uint64_t mem_data = 0;
            core_->dcache_read(&mem_data, mem_addr, vsewb);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
            setVregData(states.vtype.vsew, vreg_file, vd + f * emul, i, mem_data);
          }
        }
        break;
      }
      case 0b01000: { // vl1r.v, vl2r.v, vl4r.v, vl8r.v
        uint32_t nreg = instr.getVnf() + 1;
        if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
          std::cout << "Whole vector register load - reserved value for nreg: " << nreg << std::endl;
          std::abort();
        }

        uint32_t eew = instr.getVlsWidth() & 0x3;
        uint32_t stride = 1 << eew;
        uint32_t vl = nreg * (VLENB / vsewb);

        trace_data->vl = vl;
        trace_data->vnf = 1;

        for (uint32_t i = 0; i < vl; i++) {
          if (isMasked(vreg_file, 0, i, vmask))
            continue;
          uint64_t mem_addr = base_addr + i * stride;
          uint64_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
          setVregData(states.vtype.vsew, vreg_file, vd, i, mem_data);
        }
        break;
      }
      case 0b01011: { // vlm.v
        if (states.vtype.vsew != 0) {
          std::cout << "vlm.v only supports SEW=8, but SEW was: " << states.vtype.vsew << std::endl;
          std::abort();
        }

        uint32_t vl = (states.vl + 7) / 8;
        uint32_t stride = vsewb;

        trace_data->vl = vl;
        trace_data->vnf = 1;

        for (uint32_t i = 0; i < vl; i++) {
          if (isMasked(vreg_file, 0, i, 1))
            continue;
          uint64_t mem_addr = base_addr + i * stride;
          uint64_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
          setVregData(states.vtype.vsew, vreg_file, vd, i, mem_data);
        }
        break;
      }
      default:
        std::cout << "Load vector - unsupported lumop: " << lumop << std::endl;
        std::abort();
      }
      break;
    }
    case 0b10: {// strided: vlse8.v, vlse16.v, vlse32.v, vlse64.v
                // vlsseg2e8.v, vlsseg2e16.v, vlsseg2e32.v, vlsseg2e64.v
                // vlsseg3e8.v, vlsseg3e16.v, vlsseg3e32.v, vlsseg3e64.v
                // vlsseg4e8.v, vlsseg4e16.v, vlsseg4e32.v, vlsseg4e64.v
                // vlsseg5e8.v, vlsseg5e16.v, vlsseg5e32.v, vlsseg5e64.v
                // vlsseg6e8.v, vlsseg6e16.v, vlsseg6e32.v, vlsseg6e64.v
                // vlsseg7e8.v, vlsseg7e16.v, vlsseg7e32.v, vlsseg7e64.v
                // vlsseg8e8.v, vlsseg8e16.v, vlsseg8e32.v, vlsseg8e64.v
      uint32_t nfields = instr.getVnf() + 1;
      uint32_t emul = (states.vtype.vlmul >> 2) ? 1 : (1 << (states.vtype.vlmul & 0b11));
      assert(nfields * emul <= 8);

      WordI stride = rs2_data.at(tid).i;

      for (uint32_t i = 0; i < states.vl; i++) {
        if (isMasked(vreg_file, 0, i, vmask))
          continue;
        for (uint32_t f = 0; f < nfields; f++) {
          WordI offset = i * stride + f * vsewb;
          uint64_t mem_addr = base_addr + offset;
          uint64_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
          setVregData(states.vtype.vsew, vreg_file, vd + f * emul, i, mem_data);
        }
      }
      break;
    }
    case 0b01:  // indexed - unordered, vluxei8.v, vluxei16.v, vluxei32.v, vluxei64.v
                // vluxseg2e8.v, vluxseg2e16.v, vluxseg2e32.v, vluxseg2e64.v
                // vluxseg3e8.v, vluxseg3e16.v, vluxseg3e32.v, vluxseg3e64.v
                // vluxseg4e8.v, vluxseg4e16.v, vluxseg4e32.v, vluxseg4e64.v
                // vluxseg5e8.v, vluxseg5e16.v, vluxseg5e32.v, vluxseg5e64.v
                // vluxseg6e8.v, vluxseg6e16.v, vluxseg6e32.v, vluxseg6e64.v
                // vluxseg7e8.v, vluxseg7e16.v, vluxseg7e32.v, vluxseg7e64.v
                // vluxseg8e8.v, vluxseg8e16.v, vluxseg8e32.v, vluxseg8e64.v
    case 0b11: {// indexed - ordered, vloxei8.v, vloxei16.v, vloxei32.v, vloxei64.v
                // vloxseg2e8.v, vloxseg2e16.v, vloxseg2e32.v, vloxseg2e64.v
                // vloxseg3e8.v, vloxseg3e16.v, vloxseg3e32.v, vloxseg3e64.v
                // vloxseg4e8.v, vloxseg4e16.v, vloxseg4e32.v, vloxseg4e64.v
                // vloxseg5e8.v, vloxseg5e16.v, vloxseg5e32.v, vloxseg5e64.v
                // vloxseg6e8.v, vloxseg6e16.v, vloxseg6e32.v, vloxseg6e64.v
                // vloxseg7e8.v, vloxseg7e16.v, vloxseg7e32.v, vloxseg7e64.v
                // vloxseg8e8.v, vloxseg8e16.v, vloxseg8e32.v, vloxseg8e64.v
      uint32_t vs2 = instr.getSrcReg(1).idx;
      uint32_t nfields = instr.getVnf() + 1;
      uint32_t eew = instr.getVlsWidth() & 0x3;

      uint32_t emul = states.vtype.vlmul >> 2 ? 1 : 1 << (states.vtype.vlmul & 0b11);
      assert(nfields * emul <= 8);

      for (uint32_t i = 0; i < states.vl; i++) {
        if (isMasked(vreg_file, 0, i, vmask))
          continue;
        uint64_t offset = getVregData(eew, vreg_file, vs2, i);
        for (uint32_t f = 0; f < nfields; f++) {
          uint64_t mem_addr = base_addr + offset + f * vsewb;
          uint64_t mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
          setVregData(states.vtype.vsew, vreg_file, vd + f * emul, i, mem_data);
        }
      }
      break;
    }
    default:
      std::cout << "Load vector - unsupported mop: " << mop << std::endl;
      std::abort();
    }
  }

  void store(const Instr &instr,
             uint32_t wid,
             uint32_t tid,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs2_data,
             MemTraceData* trace_data) {
    auto& states = vpu_states_.at(wid);
    uint32_t vmask = instr.getVmask();
    uint32_t mop = instr.getVmop();
    uint32_t vsewb = 1 << states.vtype.vsew;
    auto& vreg_file = states.vreg_file.at(tid);
    uint64_t base_addr = rs1_data.at(tid).i;
    base_addr &= 0xFFFFFFFC; // TODO: riscv-tests fix

    // udpate trace data
    trace_data->vl = states.vl;
    trace_data->vnf = instr.getVnf() + 1;

    switch (mop) {
    case 0b00: { // unit-stride
      uint32_t sumop = instr.getVumop();
      switch (sumop) {
      case 0b00000: { // vse8.v, vse16.v, vse32.v, vse64.v
        uint32_t vs3 = instr.getSrcReg(1).idx;
        uint32_t nfields = instr.getVnf() + 1;
        uint32_t emul = states.vtype.vlmul >> 2 ? 1 : 1 << (states.vtype.vlmul & 0b11);
        assert(nfields * emul <= 8);

        for (uint32_t i = 0; i < states.vl; i++) {
          if (isMasked(vreg_file, 0, i, vmask))
            continue;
          for (uint32_t f = 0; f < nfields; f++) {
            uint64_t mem_addr = base_addr + (i * nfields + f) * vsewb;
            uint64_t value = getVregData(states.vtype.vsew, vreg_file, vs3 + f * emul, i);
            core_->dcache_write(&value, mem_addr, vsewb);
            trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
          }
        }
        break;
      }
      case 0b01000: { // vs1r.v, vs2r.v, vs4r.v, vs8r.v
        uint32_t nreg = instr.getVnf() + 1;
        if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
          std::cout << "Whole vector register store - reserved value for nreg: " << nreg << std::endl;
          std::abort();
        }
        uint32_t vs3 = instr.getSrcReg(1).idx;
        uint32_t stride = vsewb;
        uint32_t vl = nreg * (VLENB / vsewb);

        trace_data->vl = vl;
        trace_data->vnf = 1;

        for (uint32_t i = 0; i < vl; i++) {
          if (isMasked(vreg_file, 0, i, vmask))
            continue;
          uint64_t value = getVregData(states.vtype.vsew, vreg_file, vs3, i);
          uint64_t mem_addr = base_addr + i * stride;
          core_->dcache_write(&value, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
        }
        break;
      }
      case 0b01011: { // vsm.v
        if (states.vtype.vsew != 0) {
          std::cout << "vsm.v only supports EEW=8, but EEW was: " << states.vtype.vsew << std::endl;
          std::abort();
        }

        uint32_t vs3 = instr.getSrcReg(1).idx;
        uint32_t vl = (states.vl + 7) / 8;
        uint32_t stride = vsewb;

        trace_data->vl = vl;
        trace_data->vnf = 1;

        for (uint32_t i = 0; i < vl; i++) {
          if (isMasked(vreg_file, 0, i, 1))
            continue;
          uint64_t mem_addr = base_addr + i * stride;
          uint64_t value = getVregData(states.vtype.vsew, vreg_file, vs3, i);
          core_->dcache_write(&value, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
        }
        break;
      }
      default:
        std::cout << "Store vector - unsupported sumop: " << sumop << std::endl;
        std::abort();
      }
      break;
    }
    case 0b10: {// strided: vsse8.v, vsse16.v, vsse32.v, vsse64.v
                // vssseg2e8.v, vssseg2e16.v, vssseg2e32.v, vssseg2e64.v
                // vssseg3e8.v, vssseg3e16.v, vssseg3e32.v, vssseg3e64.v
                // vssseg4e8.v, vssseg4e16.v, vssseg4e32.v, vssseg4e64.v
                // vssseg5e8.v, vssseg5e16.v, vssseg5e32.v, vssseg5e64.v
                // vssseg6e8.v, vssseg6e16.v, vssseg6e32.v, vssseg6e64.v
                // vssseg7e8.v, vssseg7e16.v, vssseg7e32.v, vssseg7e64.v
                // vssseg8e8.v, vssseg8e16.v, vssseg8e32.v, vssseg8e64.v
      WordI stride = rs2_data.at(tid).i;
      uint32_t vs3 = instr.getSrcReg(2).idx;
      uint32_t nfields = instr.getVnf() + 1;

      uint32_t emul = states.vtype.vlmul >> 2 ? 1 : 1 << (states.vtype.vlmul & 0b11);
      assert(nfields * emul <= 8);

      for (uint32_t i = 0; i < states.vl; i++) {
        if (isMasked(vreg_file, 0, i, vmask))
          continue;
        for (uint32_t f = 0; f < nfields; f++) {
          WordI offset = i * stride + f * vsewb;
          uint64_t mem_addr = base_addr + offset;
          uint64_t value = getVregData(states.vtype.vsew, vreg_file, vs3 + f * emul, i);
          core_->dcache_write(&value, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
        }
      }
      break;
    }
    case 0b01:  // indexed - unordered, vsuxei8.v, vsuxei16.v, vsuxei32.v, vsuxei64.v
                // vsuxseg2ei8.v, vsuxseg2ei16.v, vsuxseg2ei32.v, vsuxseg2ei64.v
                // vsuxseg3ei8.v, vsuxseg3ei16.v, vsuxseg3ei32.v, vsuxseg3ei64.v
                // vsuxseg4ei8.v, vsuxseg4ei16.v, vsuxseg4ei32.v, vsuxseg4ei64.v
                // vsuxseg5ei8.v, vsuxseg5ei16.v, vsuxseg5ei32.v, vsuxseg5ei64.v
                // vsuxseg6ei8.v, vsuxseg6ei16.v, vsuxseg6ei32.v, vsuxseg6ei64.v
                // vsuxseg7ei8.v, vsuxseg7ei16.v, vsuxseg7ei32.v, vsuxseg7ei64.v
                // vsuxseg8ei8.v, vsuxseg8ei16.v, vsuxseg8ei32.v, vsuxseg8ei64.v
    case 0b11: {// indexed - ordered, vsoxei8.v, vsoxei16.v, vsoxei32.v, vsoxei64.v
                // vsoxseg2ei8.v, vsoxseg2ei16.v, vsoxseg2ei32.v, vsoxseg2ei64.v
                // vsoxseg3ei8.v, vsoxseg3ei16.v, vsoxseg3ei32.v, vsoxseg3ei64.v
                // vsoxseg4ei8.v, vsoxseg4ei16.v, vsoxseg4ei32.v, vsoxseg4ei64.v
                // vsoxseg5ei8.v, vsoxseg5ei16.v, vsoxseg5ei32.v, vsoxseg5ei64.v
                // vsoxseg6ei8.v, vsoxseg6ei16.v, vsoxseg6ei32.v, vsoxseg6ei64.v
                // vsoxseg7ei8.v, vsoxseg7ei16.v, vsoxseg7ei32.v, vsoxseg7ei64.v
                // vsoxseg8ei8.v, vsoxseg8ei16.v, vsoxseg8ei32.v, vsoxseg8ei64.v
      uint32_t vs2 = instr.getSrcReg(1).idx;
      uint32_t vs3 = instr.getSrcReg(2).idx;
      uint32_t nfields = instr.getVnf() + 1;
      uint32_t eew = instr.getVlsWidth() & 0x3;

      uint32_t emul = states.vtype.vlmul >> 2 ? 1 : 1 << (states.vtype.vlmul & 0b11);
      assert(nfields * emul <= 8);

      for (uint32_t i = 0; i < states.vl; i++) {
        if (isMasked(vreg_file, 0, i, vmask))
          continue;
        uint64_t offset = getVregData(eew, vreg_file, vs2, i);
        for (uint32_t f = 0; f < nfields; f++) {
          uint64_t mem_addr = base_addr + offset + f * vsewb;
          uint64_t value = getVregData(states.vtype.vsew, vreg_file, vs3 + f * emul, i);
          core_->dcache_write(&value, mem_addr, vsewb);
          trace_data->mem_addrs.at(tid).push_back({mem_addr, vsewb});
        }
      }
      break;
    }
    default:
      std::cout << "Store vector - unsupported mop: " << mop << std::endl;
      std::abort();
    }
  }

  ExeRet execute(const Instr &instr,
                 uint32_t wid,
                 uint32_t tid,
                 const std::vector<reg_data_t>& rs1_data,
                 const std::vector<reg_data_t>& rs2_data,
                 std::vector<reg_data_t>& rd_data,
                 ExeTraceData* trace_data) {
    auto& states = vpu_states_.at(wid);
    uint32_t funct3 = instr.getFunct3();
    uint32_t funct6 = instr.getFunct6();

    uint32_t rdest = instr.getDestReg().idx;
    uint32_t rsrc0 = instr.getSrcReg(0).idx;
    uint32_t rsrc1 = instr.getSrcReg(1).idx;
    uint32_t vmask = instr.getVmask();
    Word immsrc = sext<Word>(instr.getImm(), width_reg);
    Word uimmsrc = (Word)instr.getImm();

    auto& vreg_file = states.vreg_file.at(tid);

    bool rd_write = true; // all execute instructions writeback

    // udpate trace data
    trace_data->vl = states.vl;
    trace_data->vlmul = 1;
    // TODO: states.vlmul is not used?

    VpuType vpu_type;

    switch (funct3) {
    case 0: { // vector-vector
      vpu_type = VpuType::ARITH;
      switch (funct6) {
      case 0: { // vadd.vv
        vector_op_vv<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 2: { // vsub.vv
        vector_op_vv<Sub, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 4: { // vminu.vv
        vector_op_vv<Min, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 5: { // vmin.vv
        vector_op_vv<Min, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 6: { // vmaxu.vv
        vector_op_vv<Max, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 7: { // vmax.vv
        vector_op_vv<Max, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 9: { // vand.vv
        vector_op_vv<And, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 10: { // vor.vv
        vector_op_vv<Or, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 11: { // vxor.vv
        vector_op_vv<Xor, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 12: { // vrgather.vv
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, false, states.vlmax, vmask);
      } break;
      case 14: { // vrgatherei16.vv
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, true, states.vlmax, vmask);
      } break;
      case 16: { // vadc.vvm
        vector_op_vv_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl);
      } break;
      case 17: { // vmadc.vv, vmadc.vvm
        vector_op_vv_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 18: { // vsbc.vvm
        vector_op_vv_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl);
      } break;
      case 19: { // vmsbc.vv, vmsbc.vvm
        vector_op_vv_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 23: {
        if (vmask) { // vmv.v.v
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.v vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        } else { // vmerge.vvm
          vector_op_vv_merge<int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        }
      } break;
      case 24: { // vmseq.vv
        vector_op_vv_mask<Eq, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 25: { // vmsne.vv
        vector_op_vv_mask<Ne, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 26: { // vmsltu.vv
        vector_op_vv_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 27: { // vmslt.vv
        vector_op_vv_mask<Lt, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 28: { // vmsleu.vv
        vector_op_vv_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 29: { // vmsle.vv
        vector_op_vv_mask<Le, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 30: { // vmsgtu.vv
        vector_op_vv_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 31: { // vmsgt.vv
        vector_op_vv_mask<Gt, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 32: { // vsaddu.vv
        vector_op_vv_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 33: { // vsadd.vv
        vector_op_vv_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 34: { // vssubu.vv
        vector_op_vv_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 35: { // vssub.vv
        vector_op_vv_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 37: { // vsll.vv
        vector_op_vv<Sll, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vsmul.vv
        vector_op_vv_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 40: { // vsrl.vv
        vector_op_vv<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vsra.vv
        vector_op_vv<SrlSra, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 42: { // vssrl.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 43: { // vssra.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 44: { // vnsrl.wv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 45: { // vnsra.wv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 46: { // vnclipu.wv
        vector_op_vv_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 47: { // vnclip.wv
        vector_op_vv_n<Clip, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 48: { // vwredsumu.vs
        vector_op_vv_red_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 49: { // vwredsum.vs
        vector_op_vv_red_w<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      default:
        std::cout << "Unrecognised vector - vector instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 1: { // float vector-vector
      switch (funct6) {
      case 0: { // vfadd.vv
      vpu_type = VpuType::FMA;
        vector_op_vv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 2: { // vfsub.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 1:   // vfredusum.vs - treated the same as vfredosum.vs
      case 3: { // vfredosum.vs
        vpu_type = VpuType::FMA_R;
        vector_op_vv_red<Fadd, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 4: { // vfmin.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 5: { // vfredmin.vs
        vpu_type = VpuType::FNCP_R;
        vector_op_vv_red<Fmin, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 6: { // vfmax.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 7: { // vfredmax.vs
        vpu_type = VpuType::FNCP_R;
        vector_op_vv_red<Fmax, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 8: { // vfsgnj.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 9: { // vfsgnjn.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 10: { // vfsgnjx.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 16: { // vfmv.f.s
        vpu_type = VpuType::FNCP;
        WordI result = 0;
        vector_op_scalar(&result, vreg_file, rsrc0, rsrc1, states.vtype.vsew);
        DP(4, "Moved " << result << " from: " << +rsrc1 << " to: " << +rdest);
        rd_data.at(tid).i = result;
      } break;
      case 18: {
        vpu_type = VpuType::FCVT;
        switch (rsrc0 >> 3) {
        case 0b00: // vfcvt.xu.f.v, vfcvt.x.f.v, vfcvt.f.xu.v, vfcvt.f.x.v, vfcvt.rtz.xu.f.v, vfcvt.rtz.x.f.v
          vector_op_vix<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
          break;
        case 0b01: // vfwcvt.xu.f.v, vfwcvt.x.f.v, vfwcvt.f.xu.v, vfwcvt.f.x.v, vfwcvt.f.f.v, vfwcvt.rtz.xu.f.v, vfwcvt.rtz.x.f.v
          vector_op_vix_w<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
          break;
        case 0b10: { // vfncvt.xu.f.w, vfncvt.x.f.w, vfncvt.f.xu.w, vfncvt.f.x.w, vfncvt.f.f.w, vfncvt.rod.f.f.w, vfncvt.rtz.xu.f.w, vfncvt.rtz.x.f.w
          uint32_t vxsat = 0; // saturation argument is unused
          vector_op_vix_n<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
          break;
        }
        default:
          std::cout << "Fcvt unsupported value for rsrc0: " << rsrc0 << std::endl;
          std::abort();
        }
      } break;
      case 19: { // vfsqrt.v, vfrsqrt7.v, vfrec7.v, vfclass.v
        vpu_type = VpuType::FSQRT;
        vector_op_vix<Funary1, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 24: { // vmfeq.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv_mask<Feq, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 25: { // vmfle.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv_mask<Fle, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 27: { // vmflt.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv_mask<Flt, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 28: { // vmfne.vv
        vpu_type = VpuType::FNCP;
        vector_op_vv_mask<Fne, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 32: { // vfdiv.vv
        vpu_type = VpuType::FDIV;
        vector_op_vv<Fdiv, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 36: { // vfmul.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fmul, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 40: { // vfmadd.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vfnmadd.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 42: { // vfmsub.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 43: { // vfnmsub.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 44: { // vfmacc.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 45: { // vfnmacc.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 46: { // vfmsac.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 47: { // vfnmsac.vv
        vpu_type = VpuType::FMA;
        vector_op_vv<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 48: { // vfwadd.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 51:   // vfwredosum.vs
      case 49: { // vfwredusum.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_red_wf<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 50: { // vfwsub.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 52: { // vfwadd.wv
        vpu_type = VpuType::FMA;
        vector_op_vv_wfv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 54: { // vfwsub.wv
        vpu_type = VpuType::FMA;
        vector_op_vv_wfv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 56: { // vfwmul.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 60: { // vfwmacc.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 61: { // vfwnmacc.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 62: { // vfwmsac.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 63: { // vfwnmsac.vv
        vpu_type = VpuType::FMA;
        vector_op_vv_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      default:
        std::cout << "Unrecognised float vector - vector instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 2: { // mask vector-vector
      vpu_type = VpuType::ARITH;
      switch (funct6) {
      case 0: { // vredsum.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 1: { // vredand.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<And, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 2: { // vredor.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Or, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 3: { // vredxor.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Xor, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 4: { // vredminu.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Min, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 5: { // vredmin.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Min, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 6: { // vredmaxu.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Max, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 7: { // vredmax.vs
        vpu_type = VpuType::ARITH_R;
        vector_op_vv_red<Max, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 8: { // vaaddu.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 9: { // vaadd.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 10: { // vasubu.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 11: { // vasub.vv
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 16: { // vmv.x.s
        WordI result = 0;
        vector_op_scalar(&result, vreg_file, rsrc0, rsrc1, states.vtype.vsew);
        DP(4, "Moved " << result << " from: " << +rsrc1 << " to: " << +rdest);
        rd_data.at(tid).i = result;
      } break;
      case 18: { // vzext.vf8, vsext.vf8, vzext.vf4, vsext.vf4, vzext.vf2, vsext.vf2
        bool negativeLmul = states.vtype.vlmul >> 2;
        uint32_t illegalLmul = negativeLmul && !((8 >> (0x8 - states.vtype.vlmul)) >> (0x4 - (rsrc0 >> 1)));
        if (illegalLmul) {
          std::cout << "Lmul*vf<1/8 is not supported by vzext and vsext." << std::endl;
          std::abort();
        }
        vector_op_vix_ext<Xunary0>(rsrc0, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 20: { // vid.v
        vector_op_vid(vreg_file, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 23: { // vcompress.vm
        vector_op_vv_compress<uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl);
      } break;
      case 24: { // vmandn.mm
        vector_op_vv_mask<AndNot>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 25: { // vmand.mm
        vector_op_vv_mask<And>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 26: { // vmor.mm
        vector_op_vv_mask<Or>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 27: { // vmxor.mm
        vector_op_vv_mask<Xor>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 28: { // vmorn.mm
        vector_op_vv_mask<OrNot>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 29: { // vmnand.mm
        vector_op_vv_mask<Nand>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 30: { // vmnor.mm
        vector_op_vv_mask<Nor>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 31: { // vmxnor.mm
        vector_op_vv_mask<Xnor>(vreg_file, rsrc0, rsrc1, rdest, states.vl);
      } break;
      case 32: { // vdivu.vv
        vpu_type = VpuType::IDIV;
        vector_op_vv<Div, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 33: { // vdiv.vv
        vpu_type = VpuType::IDIV;
        vector_op_vv<Div, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 34: { // vremu.vv
        vpu_type = VpuType::IDIV;
        vector_op_vv<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 35: { // vrem.vv
        vpu_type = VpuType::IDIV;
        vector_op_vv<Rem, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 36: { // vmulhu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 37: { // vmul.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Mul, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 38: { // vmulhsu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Mulhsu, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vmulh.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Mulh, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vmadd.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Madd, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 43: { // vnmsub.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Nmsub, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 45: { // vmacc.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Macc, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 47: { // vnmsac.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv<Nmsac, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 48: { // vwaddu.vv
        vector_op_vv_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 49: { // vwadd.vv
        vector_op_vv_w<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 50: { // vwsubu.vv
        vector_op_vv_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 51: { // vwsub.vv
        vector_op_vv_w<Sub, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 52: { // vwaddu.wv
        vector_op_vv_wv<Add, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 53: { // vwadd.wv
        vector_op_vv_wv<Add, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 54: { // vwsubu.wv
        vector_op_vv_wv<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 55: { // vwsub.wv
        vector_op_vv_wv<Sub, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 56: { // vwmulu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 58: { // vwmulsu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 59: { // vwmul.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Mul, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 60: { // vwmaccu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 61: { // vwmacc.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Macc, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 63: { // vwmaccsu.vv
        vpu_type = VpuType::IMUL;
        vector_op_vv_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      default:
        std::cout << "Unrecognised mask vector - vector instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 3: { // vector-immmediate
      vpu_type = VpuType::ARITH;
      switch (funct6) {
      case 0: { // vadd.vi
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 3: { // vrsub.vi
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 9: { // vand.vi
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 10: { // vor.vi
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 11: { // vxor.vi
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 12: { // vrgather.vi
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask);
      } break;
      case 14: { // vslideup.vi
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, 0, vmask, false);
      } break;
      case 15: { // vslidedown.vi
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask, false);
      } break;
      case 16: { // vadc.vim
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl);
      } break;
      case 17: { // vmadc.vi, vmadc.vim
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 23: { // vmv.v.i
        if (vmask) { // vmv.v.i
          if (rsrc0 != 0) {
            std::cout << "For vmv.v.i vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
        } else { // vmerge.vim
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
        }
      } break;
      case 24: { // vmseq.vi
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 25: { // vmsne.vi
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 26: { // vmsltu.vi
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 27: { // vmslt.vi
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 28: { // vmsleu.vi
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 29: { // vmsle.vi
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 30: { // vmsgtu.vi
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 31: { // vmsgt.vi
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 32: { // vsaddu.vi
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 33: { // vsadd.vi
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 37: { // vsll.vi
        vector_op_vix<Sll, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vmv1r.v, vmv2r.v, vmv4r.v, vmv8r.v
        uint32_t nreg = (immsrc & 0b111) + 1;
        if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
          std::cout << "Reserved value for nreg: " << nreg << std::endl;
          std::abort();
        }
        uint32_t vl = (nreg * VLENB) >> states.vtype.vsew;
        trace_data->vl = vl;
        vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(vreg_file, rsrc0, rsrc1, rdest, states.vtype.vsew, vl, vmask);
      } break;
      case 40: { // vsrl.vi
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vsra.vi
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 42: { // vssrl.vi
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 43: { // vssra.vi
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 44: { // vnsrl.wi
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 45: { // vnsra.wi
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 46: { // vnclipu.wi
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 47: { // vnclip.wi
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(immsrc, vreg_file, rsrc0, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      default:
        std::cout << "Unrecognised vector - immidiate instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 4: { // vector-scalar
      vpu_type = VpuType::ARITH;
      auto rs1_value = rs1_data.at(tid).i;
      switch (funct6) {
      case 0: { // vadd.vx
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 2: { // vsub.vx
        vector_op_vix<Sub, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 3: { // vrsub.vx
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 4: { // vminu.vx
        vector_op_vix<Min, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 5: { // vmin.vx
        vector_op_vix<Min, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 6: { // vmaxu.vx
        vector_op_vix<Max, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 7: { // vmax.vx
        vector_op_vix<Max, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 9: { // vand.vx
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 10: { // vor.vx
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 11: { // vxor.vx
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 12: { // vrgather.vx
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask);
      } break;
      case 14: { // vslideup.vx
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, 0, vmask, false);
      } break;
      case 15: { // vslidedown.vx
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask, false);
      } break;
      case 16: { // vadc.vxm
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl);
      } break;
      case 17: { // vmadc.vx, vmadc.vxm
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 18: { // vsbc.vxm
        vector_op_vix_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl);
      } break;
      case 19: { // vmsbc.vx, vmsbc.vxm
        vector_op_vix_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 23: {
        if (vmask) { // vmv.v.x
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.x vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        } else { // vmerge.vxm
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        }
      } break;
      case 24: { // vmseq.vx
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 25: { // vmsne.vx
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 26: { // vmsltu.vx
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 27: { // vmslt.vx
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 28: { // vmsleu.vx
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 29: { // vmsle.vx
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 30: { // vmsgtu.vx
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 31: { // vmsgt.vx
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 32: { // vsaddu.vx
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 33: { // vsadd.vx
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 34: { // vssubu.vx
        vector_op_vix_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 35: { // vssub.vx
        vector_op_vix_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, states.vxsat);
      } break;
      case 37: { // vsll.vx
        vector_op_vix<Sll, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vsmul.vx
        vector_op_vix_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 40: { // vsrl.vx
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vsra.vx
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 42: { // vssrl.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 43: { // vssra.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 44: { // vnsrl.wx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 45: { // vnsra.wx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, 2, vxsat);
      } break;
      case 46: { // vnclipu.wx
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      case 47: { // vnclip.wx
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, states.vxsat);
      } break;
      default:
        std::cout << "Unrecognised vector - scalar instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 5: { // float vector-scalar
      auto rs1_value = rs1_data.at(tid).i;
      switch (funct6) {
      case 0: { // vfadd.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 2: { // vfsub.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 4: { // vfmin.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 6: { // vfmax.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 8: { // vfsgnj.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 9: { // vfsgnjn.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 10: { // vfsgnjx.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 14: { // vfslide1up.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, 0, vmask, true);
      } break;
      case 15: { // vfslide1down.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask, true);
      } break;
      case 16: { // vfmv.s.f
        vpu_type = VpuType::FNCP;
        if (rsrc1 != 0) {
          std::cout << "For vfmv.s.f vs2 must contain v0." << std::endl;
          std::abort();
        }
        uint32_t vl = std::min(states.vl, (uint32_t)1);
        trace_data->vl = vl;
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, vl, vmask);
      } break;
      case 24: { // vmfeq.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Feq, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 23: {
        vpu_type = VpuType::FNCP;
        if (vmask) { // vfmv.v.f
          if (rsrc1 != 0) {
            std::cout << "For vfmv.v.f vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        } else { // vfmerge.vfm
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
        }
      } break;
      case 25: { // vmfle.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Fle, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 27: { // vmflt.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Flt, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 28: { // vmfne.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Fne, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 29: { // vmfgt.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Fgt, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 31: { // vmfge.vf
        vpu_type = VpuType::FNCP;
        vector_op_vix_mask<Fge, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 32: { // vfdiv.vf
        vpu_type = VpuType::FDIV;
        vector_op_vix<Fdiv, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 33: { // vfrdiv.vf
        vpu_type = VpuType::FDIV;
        vector_op_vix<Frdiv, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 36: { // vfmul.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vfrsub.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Frsub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 40: { // vfmadd.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vfnmadd.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 42: { // vfmsub.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 43: { // vfnmsub.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 44: { // vfmacc.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 45: { // vfnmacc.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 46: { // vfmsac.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 47: { // vfnmsac.vf
        vpu_type = VpuType::FMA;
        vector_op_vix<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 48: { // vfwadd.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 50: { // vfwsub.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 52: { // vfwadd.wf
        vpu_type = VpuType::FMA;
        uint64_t src1_d = rv_ftod(rs1_value);
        vector_op_vix_wx<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 54: { // vfwsub.wf
        vpu_type = VpuType::FMA;
        uint64_t src1_d = rv_ftod(rs1_value);
        vector_op_vix_wx<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 56: { // vfwmul.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 60: { // vfwmacc.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 61: { // vfwnmacc.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 62: { // vfwmsac.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 63: { // vfwnmsac.vf
        vpu_type = VpuType::FMA;
        vector_op_vix_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      default:
        std::cout << "Unrecognised float vector - scalar instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 6: { // vector-scalar
      vpu_type = VpuType::ARITH;
      auto rs1_value = rs1_data.at(tid).i;
      switch (funct6) {
      case 8: { // vaaddu.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 9: { // vaadd.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 10: { // vasubu.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 11: { // vasub.vx
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask, states.vxrm, vxsat);
      } break;
      case 14: { // vslide1up.vx
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, 0, vmask, true);
      } break;
      case 15: { // vslide1down.vx
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, states.vlmax, vmask, true);
      } break;
      case 16: { // vmv.s.x
        if (rsrc1 != 0) {
          std::cout << "For vmv.s.x vs2 must contain v0." << std::endl;
          std::abort();
        }
        uint32_t vl = std::min(states.vl, (uint32_t)1);
        trace_data->vl = vl;
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, vl, vmask);
      } break;
      case 32: { // vdivu.vx
        vpu_type = VpuType::IDIV;
        vector_op_vix<Div, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 33: { // vdiv.vx
        vpu_type = VpuType::IDIV;
        vector_op_vix<Div, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 34: { // vremu.vx
        vpu_type = VpuType::IDIV;
        vector_op_vix<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 35: { // vrem.vx
        vpu_type = VpuType::IDIV;
        vector_op_vix<Rem, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 36: { // vmulhu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 37: { // vmul.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Mul, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 38: { // vmulhsu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Mulhsu, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 39: { // vmulh.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Mulh, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 41: { // vmadd.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Madd, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 43: { // vnmsub.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Nmsub, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 45: { // vmacc.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Macc, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 47: { // vnmsac.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix<Nmsac, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 48: { // vwaddu.vx
        vector_op_vix_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 49: { // vwadd.vx
        vector_op_vix_w<Add, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 50: { // vwsubu.vx
        vector_op_vix_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 51: { // vwsub.vx
        vector_op_vix_w<Sub, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 52: { // vwaddu.wx
        vector_op_vix_wx<Add, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 53: { // vwadd.wx
        uint32_t vsew_bits = 1 << (3 + states.vtype.vsew);
        Word src1_ext = sext(rs1_value, vsew_bits);
        vector_op_vix_wx<Add, int8_t, int16_t, int32_t, int64_t>(src1_ext, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 54: { // vwsubu.wx
        vector_op_vix_wx<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 55: { // vwsub.wx
        uint32_t vsew_bits = 1 << (3 + states.vtype.vsew);
        Word src1_ext = sext(rs1_value, vsew_bits);
        vector_op_vix_wx<Sub, int8_t, int16_t, int32_t, int64_t>(src1_ext, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 56: { // vwmulu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 58: { // vwmulsu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 59: { // vwmul.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Mul, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 60: { // vwmaccu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 61: { // vwmacc.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Macc, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 62: { // vwmaccus.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Maccus, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      case 63: { // vwmaccsu.vx
        vpu_type = VpuType::IMUL;
        vector_op_vix_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(rs1_value, vreg_file, rsrc1, rdest, states.vtype.vsew, states.vl, vmask);
      } break;
      default:
        std::cout << "Unrecognised vector - scalar instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
        std::abort();
      }
    } break;
    case 7: {
      vpu_type = VpuType::VSET;
      uint32_t zimm;
      if (instr.hasVattrMask(vattr_zimm)) {
        zimm = instr.getZimm();
      } else {
        zimm = rs2_data.at(tid).i;
      }

      uint32_t vlmul = zimm & mask_v_lmul;
      uint32_t vsew  = (zimm >> shift_v_sew) & mask_v_sew;
      uint32_t vta   = (zimm >> shift_v_ta) & mask_v_ta;
      uint32_t vma   = (zimm >> shift_v_ma) & mask_v_ma;

      uint32_t vlmul_neg = (vlmul >> 2);
      uint32_t vlen_mul = vlmul_neg ? (VLENB >> (8 - vlmul)) : (VLENB << vlmul);
      uint32_t vlmax = vlen_mul >> vsew;
      uint32_t vill = ((1u << vsew) > XLENB) || (vlmax > VLEN);

      uint32_t vl;
      if (instr.hasImm()) {
        // vsetivli
        vl = instr.getImm();
      } else {
        // vsetvli/vsetvl
        vl = (rsrc0 != 0) ? rs1_data.at(tid).i : ((rdest != 0) ? vlmax : states.vl);
      }

      vl = std::min(vl, vlmax);

      if (vill) {
        vl = 0;
        vma = 0;
        vta = 0;
        vsew = 0;
        vlmul = 0;
      }

      DP(4, "Vset(i)vl(i) - vill: " << vill << " vma: " << vma << " vta: " << vta << " lmul: " << vlmul << " sew: " << vsew << " vl: " << vl << " vlmax: " << vlmax);

      // update the vector unit state
      states.vstart = 0;
      states.vlmax = vlmax;
      states.vtype.vill = vill;
      states.vtype.vma = vma;
      states.vtype.vta = vta;
      states.vtype.vsew = vsew;
      states.vtype.vlmul = vlmul;
      states.vl = vl;

      // return value is the new vl
      rd_data.at(tid).i = vl;
    } break;
    default:
      std::cout << "Unrecognised vector instruction funct3: " << funct3 << " funct6: " << funct6 << std::endl;
      std::abort();
    }

    return ExeRet{vpu_type, rd_write};
  }

  bool get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value) {
    __unused (tid);
    switch (addr) {
    case VX_CSR_VSTART:
      *value = vpu_states_.at(wid).vstart;
      return true;
    case VX_CSR_VXSAT:
      *value = vpu_states_.at(wid).vxsat;
      return true;
    case VX_CSR_VXRM:
      *value = vpu_states_.at(wid).vxrm;
      return true;
    case VX_CSR_VCSR:
      *value = (vpu_states_.at(wid).vxrm << 1) |  vpu_states_.at(wid).vxsat;
      return true;
    case VX_CSR_VL:
      *value = vpu_states_.at(wid).vl;
      return true;
    case VX_CSR_VTYPE:
      *value = vpu_states_.at(wid).vtype.value;
      return true;
    case VX_CSR_VLENB:
      *value = VLENB;
      return true;
    default:
      return false;
    }
  }

  bool set_csr(uint32_t addr,uint32_t wid, uint32_t tid, Word value) {
    __unused (tid);
    switch (addr) {
    case VX_CSR_VSTART:
      vpu_states_.at(wid).vstart = value;
      return true;
    case VX_CSR_VXSAT:
      vpu_states_.at(wid).vxsat = value & 0b1;
      return true;
    case VX_CSR_VXRM:
      vpu_states_.at(wid).vxrm = value & 0b11;
      return true;
    case VX_CSR_VCSR:
      vpu_states_.at(wid).vxsat = value & 0b1;
      vpu_states_.at(wid).vxrm = (value >> 1) & 0b11;
      return true;
    case VX_CSR_VL: // read only
    case VX_CSR_VTYPE:
    case VX_CSR_VLENB:
      std::abort();
      [[fallthrough]];
    default:
      return false;
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  struct pending_req_t {
    instr_trace_t* trace;
    uint32_t count;
  };

  union vtype_t {
    struct {
      uint32_t vlmul : 3; // vector register group multiplier
      uint32_t vsew  : 3; // vector element width
      uint32_t vta   : 1; // vector tail agnostic
      uint32_t vma   : 1; // vector mask agnostic
      uint32_t reserved: 23;
      uint32_t vill  : 1; // illegal vtype
    };
    uint32_t value;
  };

  struct vpu_states_t {
    std::vector<VRF_t> vreg_file;
    uint32_t vstart;
    uint32_t vxsat;
    uint32_t vxrm;
    uint32_t vl;
    vtype_t  vtype;
    uint32_t vlenb;
    uint32_t vlmax;

    vpu_states_t(uint32_t num_threads)
      : vreg_file(num_threads, std::vector(MAX_NUM_REGS, std::vector<Byte>(VLENB, 0)))
      , vstart(0)
      , vxsat(0)
      , vxrm(0)
      , vl(0)
      , vtype({0, 0, 0, 0, 0, 0})
      , vlenb(VLENB)
      , vlmax(0)
    {}

    void reset() {
      for (auto& reg_file : this->vreg_file) {
        for (auto& reg : reg_file) {
          for (auto& elm : reg) {
          #ifndef NDEBUG
            elm = 0;
          #else
            elm = std::rand();
          #endif
          }
        }
      }
    }
  };

  VecUnit*                  simobject_;
  Core*                     core_;
  std::vector<vpu_states_t> vpu_states_;
  uint32_t                  num_lanes_;
  HashTable<pending_req_t>  pending_reqs_;
  PerfStats                 perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

VecUnit::VecUnit(const SimContext& ctx,
                 const char* name,
                 const Arch& arch,
                 Core* core)
  : SimObject<VecUnit>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
  , impl_(new Impl(this, arch, core))
{}

VecUnit::~VecUnit() {
  delete impl_;
}

void VecUnit::reset() {
  impl_->reset();
}

void VecUnit::tick() {
  impl_->tick();
}

bool VecUnit::get_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word* value) {
  return impl_->get_csr(addr, wid, tid, value);
}

bool VecUnit::set_csr(uint32_t addr, uint32_t wid, uint32_t tid, Word value) {
  return impl_->set_csr(addr, wid, tid, value);
}

void VecUnit::load(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, MemTraceData* trace_data) {
  return impl_->load(instr, wid, tid, rs1_data, rs2_data, trace_data);
}

void VecUnit::store(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, MemTraceData* trace_data) {
  return impl_->store(instr, wid, tid, rs1_data, rs2_data, trace_data);
}

VecUnit::ExeRet VecUnit::execute(const Instr &instr, uint32_t wid, uint32_t tid, const std::vector<reg_data_t>& rs1_data, const std::vector<reg_data_t>& rs2_data, std::vector<reg_data_t>& rd_data, ExeTraceData* trace_data) {
  return impl_->execute(instr, wid, tid, rs1_data, rs2_data, rd_data, trace_data);
}

const VecUnit::PerfStats& VecUnit::perf_stats() const {
  return impl_->perf_stats();
}

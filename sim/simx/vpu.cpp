// This is a fork of https://github.com/troibe/vortex/tree/simx-v2-vector
// The purpose of this fork is to make simx-v2-vector up to date with master
// Thanks to Troibe for his amazing work

#include "emulator.h"
#include "instr.h"
#include "processor_impl.h"
#include <iostream>
#include <limits>
#include <math.h>
#include <rvfloats.h>
#include <stdlib.h>
#include "vpu.h"

using namespace vortex;

void Emulator::loadVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
  auto &warp = warps_.at(wid);
  auto vmask = instr.getVmask();
  auto rdest = instr.getRDest();
  auto mop = instr.getVmop();
  switch (mop) {
  case 0b00: { // unit-stride
    auto lumop = instr.getVumop();
    switch (lumop) {
    case 0b10000:  // vle8ff.v, vle16ff.v, vle32ff.v, vle64ff.v - we do not support exceptions -> treat like regular unit stride
                   // vlseg2e8ff.v, vlseg2e16ff.v, vlseg2e32ff.v, vlseg2e64ff.v
                   // vlseg3e8ff.v, vlseg3e16ff.v, vlseg3e32ff.v, vlseg3e64ff.v
                   // vlseg4e8ff.v, vlseg4e16ff.v, vlseg4e32ff.v, vlseg4e64ff.v
                   // vlseg5e8ff.v, vlseg5e16ff.v, vlseg5e32ff.v, vlseg5e64ff.v
                   // vlseg6e8ff.v, vlseg6e16ff.v, vlseg6e32ff.v, vlseg6e64ff.v
                   // vlseg7e8ff.v, vlseg7e16ff.v, vlseg7e32ff.v, vlseg7e64ff.v
                   // vlseg8e8ff.v, vlseg8e16ff.v, vlseg8e32ff.v, vlseg8e64ff.v
    case 0b0000: { // vle8.v, vle16.v, vle32.v, vle64.v
                   // vlseg2e8.v, vlseg2e16.v, vlseg2e32.v, vlseg2e64.v
                   // vlseg3e8.v, vlseg3e16.v, vlseg3e32.v, vlseg3e64.v
                   // vlseg4e8.v, vlseg4e16.v, vlseg4e32.v, vlseg4e64.v
                   // vlseg5e8.v, vlseg5e16.v, vlseg5e32.v, vlseg5e64.v
                   // vlseg6e8.v, vlseg6e16.v, vlseg6e32.v, vlseg6e64.v
                   // vlseg7e8.v, vlseg7e16.v, vlseg7e32.v, vlseg7e64.v
                   // vlseg8e8.v, vlseg8e16.v, vlseg8e32.v, vlseg8e64.v
      WordI stride = warp.vtype.vsew / 8;
      uint32_t nfields = instr.getVnf() + 1;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, warp.vl, false, stride, nfields, warp.vtype.vlmul, vmask);
      break;
    }
    case 0b1000: { // vl1r.v, vl2r.v, vl4r.v, vl8r.v
      uint32_t nreg = instr.getVnf() + 1;
      if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
        std::cout << "Whole vector register load - reserved value for nreg: " << nreg << std::endl;
        std::abort();
      }
      DP(4, "Whole vector register load with nreg: " << nreg);
      uint32_t stride = 1 << instr.getVsew();
      uint32_t vsew_bits = stride * 8;
      uint32_t vl = nreg * VLEN / vsew_bits;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, vsew_bits, vl, false, stride, 1, 0, vmask);
      break;
    }
    case 0b1011: { // vlm.v
      if (warp.vtype.vsew != 8) {
        std::cout << "vlm.v only supports SEW=8, but SEW was: " << warp.vtype.vsew << std::endl;
        std::abort();
      }
      WordI stride = warp.vtype.vsew / 8;
      vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, (warp.vl + 7) / 8, false, stride, 1, 0, true);
      break;
    }
    default:
      std::cout << "Load vector - unsupported lumop: " << lumop << std::endl;
      std::abort();
    }
    break;
  }
  case 0b10: { // strided: vlse8.v, vlse16.v, vlse32.v, vlse64.v
               // vlsseg2e8.v, vlsseg2e16.v, vlsseg2e32.v, vlsseg2e64.v
               // vlsseg3e8.v, vlsseg3e16.v, vlsseg3e32.v, vlsseg3e64.v
               // vlsseg4e8.v, vlsseg4e16.v, vlsseg4e32.v, vlsseg4e64.v
               // vlsseg5e8.v, vlsseg5e16.v, vlsseg5e32.v, vlsseg5e64.v
               // vlsseg6e8.v, vlsseg6e16.v, vlsseg6e32.v, vlsseg6e64.v
               // vlsseg7e8.v, vlsseg7e16.v, vlsseg7e32.v, vlsseg7e64.v
               // vlsseg8e8.v, vlsseg8e16.v, vlsseg8e32.v, vlsseg8e64.v
    auto rsrc1 = instr.getRSrc(1);
    auto rdest = instr.getRDest();
    WordI stride = warp.ireg_file.at(0).at(rsrc1);
    uint32_t nfields = instr.getVnf() + 1;
    vector_op_vix_load(warp.vreg_file, this, rsdata[0][0].i, rdest, warp.vtype.vsew, warp.vl, true, stride, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  case 0b01:   // indexed - unordered, vluxei8.v, vluxei16.v, vluxei32.v, vluxei64.v
               // vluxseg2e8.v, vluxseg2e16.v, vluxseg2e32.v, vluxseg2e64.v
               // vluxseg3e8.v, vluxseg3e16.v, vluxseg3e32.v, vluxseg3e64.v
               // vluxseg4e8.v, vluxseg4e16.v, vluxseg4e32.v, vluxseg4e64.v
               // vluxseg5e8.v, vluxseg5e16.v, vluxseg5e32.v, vluxseg5e64.v
               // vluxseg6e8.v, vluxseg6e16.v, vluxseg6e32.v, vluxseg6e64.v
               // vluxseg7e8.v, vluxseg7e16.v, vluxseg7e32.v, vluxseg7e64.v
               // vluxseg8e8.v, vluxseg8e16.v, vluxseg8e32.v, vluxseg8e64.v
  case 0b11: { // indexed - ordered, vloxei8.v, vloxei16.v, vloxei32.v, vloxei64.v
               // vloxseg2e8.v, vloxseg2e16.v, vloxseg2e32.v, vloxseg2e64.v
               // vloxseg3e8.v, vloxseg3e16.v, vloxseg3e32.v, vloxseg3e64.v
               // vloxseg4e8.v, vloxseg4e16.v, vloxseg4e32.v, vloxseg4e64.v
               // vloxseg5e8.v, vloxseg5e16.v, vloxseg5e32.v, vloxseg5e64.v
               // vloxseg6e8.v, vloxseg6e16.v, vloxseg6e32.v, vloxseg6e64.v
               // vloxseg7e8.v, vloxseg7e16.v, vloxseg7e32.v, vloxseg7e64.v
               // vloxseg8e8.v, vloxseg8e16.v, vloxseg8e32.v, vloxseg8e64.v
    uint32_t nfields = instr.getVnf() + 1;
    uint32_t vsew_bits = 1 << (3 + instr.getVsew());
    vector_op_vv_load(warp.vreg_file, this, rsdata[0][0].i, instr.getRSrc(1), rdest, warp.vtype.vsew, vsew_bits, warp.vl, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  default:
    std::cout << "Load vector - unsupported mop: " << mop << std::endl;
    std::abort();
  }
}

void Emulator::storeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
  auto &warp = warps_.at(wid);
  auto vmask = instr.getVmask();
  auto mop = instr.getVmop();
  switch (mop) {
  case 0b00: { // unit-stride
    auto vs3 = instr.getRSrc(1);
    auto sumop = instr.getVumop();
    WordI stride = warp.vtype.vsew / 8;
    switch (sumop) {
    case 0b0000: { // vse8.v, vse16.v, vse32.v, vse64.v
      uint32_t nfields = instr.getVnf() + 1;
      vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, warp.vl, false, stride, nfields, warp.vtype.vlmul, vmask);
      break;
    }
    case 0b1000: { // vs1r.v, vs2r.v, vs4r.v, vs8r.v
      uint32_t nreg = instr.getVnf() + 1;
      if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
        std::cout << "Whole vector register store - reserved value for nreg: " << nreg << std::endl;
        std::abort();
      }
      DP(4, "Whole vector register store with nreg: " << nreg);
      uint32_t vl = nreg * VLEN / 8;
      vector_op_vix_store<uint8_t>(warp.vreg_file, this, rsdata[0][0].i, vs3, vl, false, stride, 1, 0, vmask);
      break;
    }
    case 0b1011: { // vsm.v
      if (warp.vtype.vsew != 8) {
        std::cout << "vsm.v only supports EEW=8, but EEW was: " << warp.vtype.vsew << std::endl;
        std::abort();
      }
      vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, (warp.vl + 7) / 8, false, stride, 1, 0, true);
      break;
    }
    default:
      std::cout << "Store vector - unsupported sumop: " << sumop << std::endl;
      std::abort();
    }
    break;
  }
  case 0b10: { // strided: vsse8.v, vsse16.v, vsse32.v, vsse64.v
               // vssseg2e8.v, vssseg2e16.v, vssseg2e32.v, vssseg2e64.v
               // vssseg3e8.v, vssseg3e16.v, vssseg3e32.v, vssseg3e64.v
               // vssseg4e8.v, vssseg4e16.v, vssseg4e32.v, vssseg4e64.v
               // vssseg5e8.v, vssseg5e16.v, vssseg5e32.v, vssseg5e64.v
               // vssseg6e8.v, vssseg6e16.v, vssseg6e32.v, vssseg6e64.v
               // vssseg7e8.v, vssseg7e16.v, vssseg7e32.v, vssseg7e64.v
               // vssseg8e8.v, vssseg8e16.v, vssseg8e32.v, vssseg8e64.v
    auto rsrc1 = instr.getRSrc(1);
    auto vs3 = instr.getRSrc(2);
    WordI stride = warp.ireg_file.at(0).at(rsrc1);
    uint32_t nfields = instr.getVnf() + 1;
    vector_op_vix_store(warp.vreg_file, this, rsdata[0][0].i, vs3, warp.vtype.vsew, warp.vl, true, stride, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  case 0b01:   // indexed - unordered, vsuxei8.v, vsuxei16.v, vsuxei32.v, vsuxei64.v
               // vsuxseg2ei8.v, vsuxseg2ei16.v, vsuxseg2ei32.v, vsuxseg2ei64.v
               // vsuxseg3ei8.v, vsuxseg3ei16.v, vsuxseg3ei32.v, vsuxseg3ei64.v
               // vsuxseg4ei8.v, vsuxseg4ei16.v, vsuxseg4ei32.v, vsuxseg4ei64.v
               // vsuxseg5ei8.v, vsuxseg5ei16.v, vsuxseg5ei32.v, vsuxseg5ei64.v
               // vsuxseg6ei8.v, vsuxseg6ei16.v, vsuxseg6ei32.v, vsuxseg6ei64.v
               // vsuxseg7ei8.v, vsuxseg7ei16.v, vsuxseg7ei32.v, vsuxseg7ei64.v
               // vsuxseg8ei8.v, vsuxseg8ei16.v, vsuxseg8ei32.v, vsuxseg8ei64.v
  case 0b11: { // indexed - ordered, vsoxei8.v, vsoxei16.v, vsoxei32.v, vsoxei64.v
               // vsoxseg2ei8.v, vsoxseg2ei16.v, vsoxseg2ei32.v, vsoxseg2ei64.v
               // vsoxseg3ei8.v, vsoxseg3ei16.v, vsoxseg3ei32.v, vsoxseg3ei64.v
               // vsoxseg4ei8.v, vsoxseg4ei16.v, vsoxseg4ei32.v, vsoxseg4ei64.v
               // vsoxseg5ei8.v, vsoxseg5ei16.v, vsoxseg5ei32.v, vsoxseg5ei64.v
               // vsoxseg6ei8.v, vsoxseg6ei16.v, vsoxseg6ei32.v, vsoxseg6ei64.v
               // vsoxseg7ei8.v, vsoxseg7ei16.v, vsoxseg7ei32.v, vsoxseg7ei64.v
               // vsoxseg8ei8.v, vsoxseg8ei16.v, vsoxseg8ei32.v, vsoxseg8ei64.v
    uint32_t nfields = instr.getVnf() + 1;
    uint32_t vsew_bits = 1 << (3 + instr.getVsew());
    vector_op_vv_store(warp.vreg_file, this, rsdata[0][0].i, instr.getRSrc(1), instr.getRSrc(2), warp.vtype.vsew, vsew_bits, warp.vl, nfields, warp.vtype.vlmul, vmask);
    break;
  }
  default:
    std::cout << "Store vector - unsupported mop: " << mop << std::endl;
    std::abort();
  }
}

void Emulator::executeVector(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata) {
  auto &warp = warps_.at(wid);
  auto func3 = instr.getFunc3();
  auto func6 = instr.getFunc6();

  auto rdest = instr.getRDest();
  auto rsrc0 = instr.getRSrc(0);
  auto rsrc1 = instr.getRSrc(1);
  auto immsrc = sext((Word)instr.getImm(), width_reg);
  auto uimmsrc = (Word)instr.getImm();
  auto vmask = instr.getVmask();
  auto num_threads = arch_.num_threads();

  switch (func3) {
  case 0: { // vector - vector
    switch (func6) {
    case 0: { // vadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vminu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Min, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vmin.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Min, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vmaxu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Max, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vmax.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Max, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<And, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Or, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Xor, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, false, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vrgatherei16.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_gather<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, true, warp.vlmax, vmask);
      }
    } break;
    case 16: { // vadc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vv, vmadc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 18: { // vsbc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 19: { // vmsbc.vv, vmsbc.vvm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.v
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.v vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vvm
          vector_op_vv_merge<int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Eq, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Ne, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Lt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Le, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Gt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 34: { // vssubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 35: { // vssub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Sll, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vsmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 40: { // vsrl.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vv_n<Clip, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 48: { // vwredsumu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwredsum.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_w<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 1: { // float vector - vector
    switch (func6) {
    case 0: { // vfadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vfsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 1:   // vfredusum.vs - treated the same as vfredosum.vs
    case 3: { // vfredosum.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fadd, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vfmin.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vfredmin.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fmin, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vfmax.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vfredmax.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Fmax, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vfsgnj.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vfsgnjn.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vfsgnjx.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 16: { // vfmv.f.s
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &dest = rddata[t].u64;
        vector_op_scalar(dest, warp.vreg_file, rsrc0, rsrc1, warp.vtype.vsew);
        DP(4, "Moved " << +dest << " from: " << +rsrc1 << " to: " << +rdest);
      }
    } break;
    case 18: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        switch (rsrc0 >> 3) {
        case 0b00: // vfcvt.xu.f.v, vfcvt.x.f.v, vfcvt.f.xu.v, vfcvt.f.x.v, vfcvt.rtz.xu.f.v, vfcvt.rtz.x.f.v
          vector_op_vix<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
          break;
        case 0b01: // vfwcvt.xu.f.v, vfwcvt.x.f.v, vfwcvt.f.xu.v, vfwcvt.f.x.v, vfwcvt.f.f.v, vfwcvt.rtz.xu.f.v, vfwcvt.rtz.x.f.v
          vector_op_vix_w<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
          break;
        case 0b10: { // vfncvt.xu.f.w, vfncvt.x.f.w, vfncvt.f.xu.w, vfncvt.f.x.w, vfncvt.f.f.w, vfncvt.rod.f.f.w, vfncvt.rtz.xu.f.w, vfncvt.rtz.x.f.w
          uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
          uint32_t vxsat = 0; // saturation argument is unused
          vector_op_vix_n<Fcvt, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
          break;
        }
        default:
          std::cout << "Fcvt unsupported value for rsrc0: " << rsrc0 << std::endl;
          std::abort();
        }
      }
    } break;
    case 19: { // vfsqrt.v, vfrsqrt7.v, vfrec7.v, vfclass.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Funary1, uint8_t, uint16_t, uint32_t, uint64_t>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 24: { // vmfeq.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Feq, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmfle.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Fle, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmflt.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Flt, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmfne.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Fne, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vfdiv.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fdiv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vfmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 40: { // vfmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vfnmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vfmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vfnmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 44: { // vfmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vfnmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 46: { // vfmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vfnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vfwadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51:   // vfwredosum.vs - treated the same as vfwredosum.vs
    case 49: { // vfwredusum.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red_wf<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vfwsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vfwadd.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wfv<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vfwsub.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wfv<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vfwmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vfwmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vfwnmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vfwmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vfwnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised float vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 2: { // mask vector - vector
    switch (func6) {
    case 0: { // vredsum.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 1: { // vredand.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<And, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vredor.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Or, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vredxor.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Xor, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vredminu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Min, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vredmin.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Min, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vredmaxu.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Max, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vredmax.vs
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_red<Max, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vaaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 9: { // vaadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 10: { // vasubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 11: { // vasub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vv_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 16: { // vmv.x.s
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &dest = rddata[t].i;
        vector_op_scalar(dest, warp.vreg_file, rsrc0, rsrc1, warp.vtype.vsew);
        DP(4, "Moved " << +dest << " from: " << +rsrc1 << " to: " << +rdest);
      }
    } break;
    case 18: { // vzext.vf8, vsext.vf8, vzext.vf4, vsext.vf4, vzext.vf2, vsext.vf2
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        bool negativeLmul = warp.vtype.vlmul >> 2;
        uint32_t illegalLmul = negativeLmul && !((8 >> (0x8 - warp.vtype.vlmul)) >> (0x4 - (rsrc0 >> 1)));
        if (illegalLmul) {
          std::cout << "Lmul*vf<1/8 is not supported by vzext and vsext." << std::endl;
          std::abort();
        }
        vector_op_vix_ext<Xunary0>(rsrc0, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 20: { // vid.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vid(warp.vreg_file, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: { // vcompress.vm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_compress<uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 24: { // vmandn.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<AndNot>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 25: { // vmand.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<And>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 26: { // vmor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Or>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 27: { // vmxor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Xor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 28: { // vmorn.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<OrNot>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 29: { // vmnand.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Nand>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 30: { // vmnor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Nor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 31: { // vmxnor.mm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_mask<Xnor>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vl);
      }
    } break;
    case 32: { // vdivu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Div, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vdiv.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Div, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 34: { // vremu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 35: { // vrem.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Rem, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vmulhu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 37: { // vmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 38: { // vmulhsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulhsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmulh.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mulh, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vmadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Madd, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vnmsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Nmsub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Macc, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vnmsac.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Nmsac, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vwaddu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwadd.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vwsubu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51: { // vwsub.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vwaddu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Add, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 53: { // vwadd.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Add, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vwsubu.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 55: { // vwsub.wv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_wv<Sub, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vwmulu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 58: { // vwmulsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 59: { // vwmul.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Mul, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vwmaccu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vwmacc.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Macc, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vwmaccsu.vv
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised mask vector - vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 3: { // vector - immidiate
    switch (func6) {
    case 0: { // vadd.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vrsub.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vslideup.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, 0, vmask, false);
      }
    } break;
    case 15: { // vslidedown.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(uimmsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, false);
      }
    } break;
    case 16: { // vadc.vim
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vi, vmadc.vim
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: { // vmv.v.i
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.i
          if (rsrc0 != 0) {
            std::cout << "For vmv.v.i vs2 must contain v0." << std::endl;
            std::abort();
          }
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vim
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<Sll, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmv1r.v, vmv2r.v, vmv4r.v, vmv8r.v
      for (uint32_t t = 0; t < num_threads; ++t) {
        uint32_t nreg = (immsrc & 0b111) + 1;
        if (nreg != 1 && nreg != 2 && nreg != 4 && nreg != 8) {
          std::cout << "Reserved value for nreg: " << nreg << std::endl;
          std::abort();
        }
        if (!warp.tmask.test(t))
          continue;
        vector_op_vv<Mv, int8_t, int16_t, int32_t, int64_t>(warp.vreg_file, rsrc0, rsrc1, rdest, warp.vtype.vsew, nreg * VLEN / warp.vtype.vsew, vmask);
      }
    } break;
    case 40: { // vsrl.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wi
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(immsrc, warp.vreg_file, rsrc0, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    default:
      std::cout << "Unrecognised vector - immidiate instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 4: {
    switch (func6) {
    case 0: { // vadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Add, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Sub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 3: { // vrsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rsub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vminu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Min, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 5: { // vmin.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Min, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vmaxu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Max, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 7: { // vmax.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Max, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vand.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<And, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vor.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Or, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 11: { // vxor.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Xor, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 12: { // vrgather.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_gather<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask);
      }
    } break;
    case 14: { // vslideup.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, false);
      }
    } break;
    case 15: { // vslidedown.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, false);
      }
    } break;
    case 16: { // vadc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry<Adc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 17: { // vmadc.vx, vmadc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry_out<Madc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 18: { // vsbc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry<Sbc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl);
      }
    } break;
    case 19: { // vmsbc.vx, vmsbc.vxm
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_carry_out<Msbc, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vmv.v.x
          if (rsrc1 != 0) {
            std::cout << "For vmv.v.x vs2 must contain v0." << std::endl;
            std::abort();
          }
          auto &src1 = warp.ireg_file.at(t).at(rsrc0);
          vector_op_vix<Mv, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vmerge.vxm
          auto &src1 = warp.ireg_file.at(t).at(rsrc0);
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 24: { // vmseq.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Eq, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 25: { // vmsne.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Ne, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 26: { // vmsltu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Lt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmslt.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Lt, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmsleu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Le, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmsle.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Le, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 30: { // vmsgtu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Gt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmsgt.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Gt, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vsaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 33: { // vsadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Sadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 34: { // vssubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Ssubu, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 35: { // vssub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Ssub, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 37: { // vsll.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Sll, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vsmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_sat<Smul, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 40: { // vsrl.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vsra.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vssrl.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_scale<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 43: { // vssra.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_scale<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 44: { // vnsrl.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 45: { // vnsra.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_n<SrlSra, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, 2, vxsat);
      }
    } break;
    case 46: { // vnclipu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    case 47: { // vnclip.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = this->get_csr(VX_CSR_VXSAT, t, wid);
        vector_op_vix_n<Clip, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
        this->set_csr(VX_CSR_VXSAT, vxsat, t, wid);
      }
    } break;
    default:
      std::cout << "Unrecognised vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 5: { // float vector - scalar
    switch (func6) {
    case 0: { // vfadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 2: { // vfsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 4: { // vfmin.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmin, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 6: { // vfmax.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmax, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 8: { // vfsgnj.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnj, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 9: { // vfsgnjn.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnjn, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 10: { // vfsgnjx.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fsgnjx, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 14: { // vfslide1up.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, true);
      }
    } break;
    case 15: { // vfslide1down.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, true);
      }
    } break;
    case 16: { // vfmv.s.f
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (rsrc1 != 0) {
          std::cout << "For vfmv.s.f vs2 must contain v0." << std::endl;
          std::abort();
        }
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, std::min(warp.vl, (uint32_t)1), vmask);
      }
    } break;
    case 24: { // vmfeq.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Feq, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 23: {
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (vmask) { // vfmv.v.f
          if (rsrc1 != 0) {
            std::cout << "For vfmv.v.f vs2 must contain v0." << std::endl;
            std::abort();
          }
          auto &src1 = warp.freg_file.at(t).at(rsrc0);
          vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        } else { // vfmerge.vfm
          auto &src1 = warp.freg_file.at(t).at(rsrc0);
          vector_op_vix_merge<int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
        }
      }
    } break;
    case 25: { // vmfle.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fle, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 27: { // vmflt.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Flt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 28: { // vmfne.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fne, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 29: { // vmfgt.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fgt, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 31: { // vmfge.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_mask<Fge, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 32: { // vfdiv.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fdiv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vfrdiv.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Frdiv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vfmul.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vfrsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Frsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 40: { // vfmadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vfnmadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 42: { // vfmsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vfnmsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 44: { // vfmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vfnmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 46: { // vfmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vfnmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vfwadd.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vfwsub.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vfwadd.wf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        uint64_t src1_d = rv_ftod(src1);
        vector_op_vix_wx<Fadd, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vfwsub.wf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        uint64_t src1_d = rv_ftod(src1);
        vector_op_vix_wx<Fsub, uint8_t, uint16_t, uint32_t, uint64_t>(src1_d, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vfwmul.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vfwmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vfwnmacc.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fnmacc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vfwmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vfwnmsac.vf
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.freg_file.at(t).at(rsrc0);
        vector_op_vix_w<Fnmsac, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised float vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 6: {
    switch (func6) {
    case 8: { // vaaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 9: { // vaadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Aadd, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 10: { // vasubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 11: { // vasub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        uint32_t vxrm = this->get_csr(VX_CSR_VXRM, t, wid);
        uint32_t vxsat = 0; // saturation is not relevant for this operation
        vector_op_vix_sat<Asub, int8_t, int16_t, int32_t, int64_t, __int128_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask, vxrm, vxsat);
      }
    } break;
    case 14: { // vslide1up.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, 0, vmask, true);
      }
    } break;
    case 15: { // vslide1down.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_slide<uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, warp.vlmax, vmask, true);
      }
    } break;
    case 16: { // vmv.s.x
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        if (rsrc1 != 0) {
          std::cout << "For vmv.s.x vs2 must contain v0." << std::endl;
          std::abort();
        }
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mv, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, std::min(warp.vl, (uint32_t)1), vmask);
      }
    } break;
    case 32: { // vdivu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Div, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 33: { // vdiv.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Div, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 34: { // vremu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rem, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 35: { // vrem.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Rem, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 36: { // vmulhu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulhu, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 37: { // vmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mul, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 38: { // vmulhsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulhsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 39: { // vmulh.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Mulh, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 41: { // vmadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Madd, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 43: { // vnmsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Nmsub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 45: { // vmacc.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Macc, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 47: { // vnmsac.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix<Nmsac, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 48: { // vwaddu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Add, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 49: { // vwadd.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Add, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 50: { // vwsubu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 51: { // vwsub.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Sub, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 52: { // vwaddu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_wx<Add, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 53: { // vwadd.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        Word src1_ext = sext(src1, warp.vtype.vsew);
        vector_op_vix_wx<Add, int8_t, int16_t, int32_t, int64_t>(src1_ext, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 54: { // vwsubu.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_wx<Sub, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 55: { // vwsub.wx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        Word &src1 = warp.ireg_file.at(t).at(rsrc0);
        Word src1_ext = sext(src1, warp.vtype.vsew);
        vector_op_vix_wx<Sub, int8_t, int16_t, int32_t, int64_t>(src1_ext, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 56: { // vwmulu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mul, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 58: { // vwmulsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mulsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 59: { // vwmul.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Mul, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 60: { // vwmaccu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Macc, uint8_t, uint16_t, uint32_t, uint64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 61: { // vwmacc.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Macc, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 62: { // vwmaccus.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Maccus, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    case 63: { // vwmaccsu.vx
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto &src1 = warp.ireg_file.at(t).at(rsrc0);
        vector_op_vix_w<Maccsu, int8_t, int16_t, int32_t, int64_t>(src1, warp.vreg_file, rsrc1, rdest, warp.vtype.vsew, warp.vl, vmask);
      }
    } break;
    default:
      std::cout << "Unrecognised vector - scalar instruction func3: " << func3 << " func6: " << func6 << std::endl;
      std::abort();
    }
  } break;
  case 7: {
    uint32_t vma = instr.getVma();
    uint32_t vta = instr.getVta();
    uint32_t vsew = instr.getVsew();
    uint32_t vlmul = instr.getVlmul();

    if (!instr.hasZimm()) { // vsetvl
      uint32_t zimm = rsdata[0][1].u;
      vlmul = zimm & mask_v_lmul;
      vsew = (zimm >> shift_v_sew) & mask_v_sew;
      vta = (zimm >> shift_v_ta) & mask_v_ta;
      vma = (zimm >> shift_v_ma) & mask_v_ma;
    }

    bool negativeLmul = vlmul >> 2;
    uint32_t vlenDividedByLmul = VLEN >> (0x8 - vlmul);
    uint32_t vlenMultipliedByLmul = VLEN << vlmul;
    uint32_t vlenTimesLmul = negativeLmul ? vlenDividedByLmul : vlenMultipliedByLmul;
    uint32_t vsew_bits = 1 << (3 + vsew);
    warp.vlmax = vlenTimesLmul / vsew_bits;
    warp.vtype.vill = (vsew_bits > XLEN) || (warp.vlmax < (VLEN / XLEN));

    Word s0 = instr.getImm(); // vsetivli
    if (!instr.hasImm()) {    // vsetvli/vsetvl
      s0 = rsdata[0][0].u;
    }

    DP(4, "Vset(i)vl(i) - vill: " << +warp.vtype.vill << " vma: " << vma << " vta: " << vta << " lmul: " << vlmul << " sew: " << vsew << " s0: " << s0 << " vlmax: " << warp.vlmax);
    warp.vl = std::min(s0, warp.vlmax);

    if (warp.vtype.vill) {
      this->set_csr(VX_CSR_VTYPE, (Word)1 << (XLEN - 1), 0, wid);
      warp.vtype.vma = 0;
      warp.vtype.vta = 0;
      warp.vtype.vsew = 0;
      warp.vtype.vlmul = 0;
      this->set_csr(VX_CSR_VL, 0, 0, wid);
      rddata[0].i = warp.vl;
    } else {
      warp.vtype.vma = vma;
      warp.vtype.vta = vta;
      warp.vtype.vsew = vsew_bits;
      warp.vtype.vlmul = vlmul;
      Word vtype_ = vlmul;
      vtype_ |= vsew << shift_v_sew;
      vtype_ |= vta << shift_v_ta;
      vtype_ |= vma << shift_v_ma;
      this->set_csr(VX_CSR_VTYPE, vtype_, 0, wid);
      this->set_csr(VX_CSR_VL, warp.vl, 0, wid);
      rddata[0].i = warp.vl;
    }
  }
    this->set_csr(VX_CSR_VSTART, 0, 0, wid);
    break;
  default:
    std::cout << "Unrecognised vector instruction func3: " << func3 << " func6: " << func6 << std::endl;
    std::abort();
  }
}

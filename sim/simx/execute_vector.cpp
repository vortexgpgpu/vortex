#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "warp.h"
#include "instr.h"
#include "core.h"
#include "execute.h"

using namespace vortex;

void executeVector(const Instr &instr, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata, std::vector<std::vector<Byte>> &vreg_file_, vtype vtype_, uint32_t vl_) {
  auto func3  = instr.getFunc3();
  auto func6  = instr.getFunc6();

  auto rdest  = instr.getRDest();
  auto rsrc0  = instr.getRSrc(0);
  auto rsrc1  = instr.getRSrc(1);
  auto vmask  = instr.getVmask();
  
    uint32_t VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
    switch (func3) {
    case 0: // vector-vector
      switch (func6) {
      case 0: {
        auto& vr1 = vreg_file_.at(rsrc0);
        auto& vr2 = vreg_file_.at(rsrc1);
        auto& vd = vreg_file_.at(rdest);
        auto& mask = vreg_file_.at(0);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t emask = *(uint8_t *)(mask.data() + i);
            uint8_t value = emask & 0x1;
            if (vmask || (!vmask && value)) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = first + second;
              DP(3, "Adding " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t emask = *(uint16_t *)(mask.data() + i);
            uint16_t value = emask & 0x1;
            if (vmask || (!vmask && value)) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = first + second;
              DP(3, "Adding " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t emask = *(uint32_t *)(mask.data() + i);
            uint32_t value = emask & 0x1;
            if (vmask || (!vmask && value)) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = first + second;
              DP(3, "Adding " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        }                
      } break;
      case 24: {
        // vmseq
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 25: { 
        // vmsne
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 26: {
        // vmsltu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 27: {
        // vmslt
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 28: {
        // vmsleu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 29: {
        // vmsle
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 30: {
        // vmsgtu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 31: {
        // vmsgt
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      }
      break;
    case 2: {
      switch (func6) {
      case 24: { 
        // vmandnot
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }            
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 25: {
        // vmand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 26: {
        // vmor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 27: { 
        // vmxor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 28: {
        // vmornot
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 29: {
        // vmnand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 30: {
        // vmnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 31: {
        // vmxnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 45: {
        // vmacc
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      }
    } break;
    case 6: {
      switch (func6) {
      case 0: {
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul.vx
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      }
    } break;
    case 7: {
      vtype_.vill  = 0;
      vtype_.vediv = instr.getVediv();
      vtype_.vsew  = instr.getVsew();
      vtype_.vlmul = instr.getVlmul();

      DP(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << rsdata[0][0].i << "VLMAX" << VLMAX);

      auto s0 = rsdata[0][0].u;
      if (s0 <= VLMAX) {
        vl_ = s0;
      } else if (s0 < (2 * VLMAX)) {
        vl_ = (uint32_t)ceil((s0 * 1.0) / 2.0);
      } else if (s0 >= (2 * VLMAX)) {
        vl_ = VLMAX;
      }        
      rddata[0].i = vl_;
    } break;
    default:
      std::abort();
    }
}
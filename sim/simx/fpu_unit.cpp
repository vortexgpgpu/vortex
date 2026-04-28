// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fpu_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include <rvfloats.h>
#include "debug.h"
#include "core.h"
#include "csr_unit.h"
#include "constants.h"

using namespace vortex;

namespace {
inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}
inline bool is_nan_boxed(uint64_t value) {
  return (uint32_t(value >> 32) == 0xffffffff);
}
inline int64_t check_boxing(int64_t a) {
  if (is_nan_boxed(a))
    return a;
  return nan_box(0x7fc00000); // NaN
}
}

FpuUnit::FpuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<NUM_FPU_BLOCKS>(ctx, name, core)
{}

uint32_t FpuUnit::latency_of(const instr_trace_t* trace) const {
	auto fpu_type = std::get<FpuType>(trace->op_type);
	const uint32_t delay = 2;
	switch (fpu_type) {
	case FpuType::FCMP:
	case FpuType::FSGNJ:
	case FpuType::FCLASS:
	case FpuType::FMVXW:
	case FpuType::FMVWX:
	case FpuType::FMINMAX:
		return 2+delay;
	case FpuType::FADD:
	case FpuType::FSUB:
	case FpuType::FMUL:
	case FpuType::FMADD:
	case FpuType::FMSUB:
	case FpuType::FNMADD:
	case FpuType::FNMSUB:
		return LATENCY_FMA+delay;
	case FpuType::FDIV:
		return LATENCY_FDIV+delay;
	case FpuType::FSQRT:
		return LATENCY_FSQRT+delay;
	case FpuType::F2I:
	case FpuType::I2F:
	case FpuType::F2F:
		return LATENCY_FCVT+delay;
	default:
		std::abort();
	}
}

void FpuUnit::execute(instr_trace_t* trace) {
	auto& emu = core_->csr_unit();           // CSR/FCSR helpers
	// Use trace->tmask captured at issue, not the live warp.tmask. Divergent
	// control flow may change the warp's tmask before this trace executes;
	// see LsuUnit::execute for the same fix.
	auto& tmask = trace->tmask;
	auto& instr = *trace->instr_ptr;
	auto instrArgs = instr.get_args();
	auto fpuArgs = std::get<IntrFpuArgs>(instrArgs);
	auto fpu_type = std::get<FpuType>(trace->op_type);
	uint32_t wid = trace->wid;
	uint32_t num_threads = NUM_THREADS;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];
	auto& rs3_data = trace->src_data[2];

	uint32_t thread_start = 0;
	for (; thread_start < num_threads; ++thread_start) {
		if (tmask.test(thread_start)) break;
	}

	trace->dst_data.assign(num_threads, reg_data_t{});
	auto& rd_data = trace->dst_data;

	switch (fpu_type) {
	case FpuType::FADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fadd_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fsub_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMUL: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmul_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmul_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FDIV: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fdiv_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fdiv_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSQRT: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fsqrt_d(rs1_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rs1_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSGNJ: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].u64 = rv_fsgnj_d(rs1_data[t].u64, rs2_data[t].u64); break;
				case 1: rd_data[t].u64 = rv_fsgnjn_d(rs1_data[t].u64, rs2_data[t].u64); break;
				case 2: rd_data[t].u64 = rv_fsgnjx_d(rs1_data[t].u64, rs2_data[t].u64); break;
				}
			} else {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].u64 = nan_box(rv_fsgnj_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				case 1: rd_data[t].u64 = nan_box(rv_fsgnjn_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				case 2: rd_data[t].u64 = nan_box(rv_fsgnjx_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMINMAX: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				if (fpuArgs.frm) rd_data[t].u64 = rv_fmax_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
				else             rd_data[t].u64 = rv_fmin_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
			} else {
				if (fpuArgs.frm) rd_data[t].u64 = nan_box(rv_fmax_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
				else             rd_data[t].u64 = nan_box(rv_fmin_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FCMP: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].i = rv_fle_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				case 1: rd_data[t].i = rv_flt_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				case 2: rd_data[t].i = rv_feq_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				}
			} else {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].i = rv_fle_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				case 1: rd_data[t].i = rv_flt_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				case 2: rd_data[t].i = rv_feq_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::F2I: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].i = sext((uint64_t)rv_ftoi_d(rs1_data[t].u64, frm, &fflags), 32); break;
				case 1: rd_data[t].i = sext((uint64_t)rv_ftou_d(rs1_data[t].u64, frm, &fflags), 32); break;
				case 2: rd_data[t].i = rv_ftol_d(rs1_data[t].u64, frm, &fflags); break;
				case 3: rd_data[t].i = rv_ftolu_d(rs1_data[t].u64, frm, &fflags); break;
				}
			} else {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].i = sext((uint64_t)rv_ftoi_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32); break;
				case 1: rd_data[t].i = sext((uint64_t)rv_ftou_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32); break;
				case 2: rd_data[t].i = rv_ftol_s(check_boxing(rs1_data[t].u64), frm, &fflags); break;
				case 3: rd_data[t].i = rv_ftolu_s(check_boxing(rs1_data[t].u64), frm, &fflags); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::I2F: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].u64 = rv_itof_d(rs1_data[t].i, frm, &fflags); break;
				case 1: rd_data[t].u64 = rv_utof_d(rs1_data[t].i, frm, &fflags); break;
				case 2: rd_data[t].u64 = rv_ltof_d(rs1_data[t].i, frm, &fflags); break;
				case 3: rd_data[t].u64 = rv_lutof_d(rs1_data[t].i, frm, &fflags); break;
				}
			} else {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].u64 = nan_box(rv_itof_s(rs1_data[t].i, frm, &fflags)); break;
				case 1: rd_data[t].u64 = nan_box(rv_utof_s(rs1_data[t].i, frm, &fflags)); break;
				case 2: rd_data[t].u64 = nan_box(rv_ltof_s(rs1_data[t].i, frm, &fflags)); break;
				case 3: rd_data[t].u64 = nan_box(rv_lutof_s(rs1_data[t].i, frm, &fflags)); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::F2F: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_ftod(check_boxing(rs1_data[t].u64), frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_dtof(rs1_data[t].u64, frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FCLASS: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].i = rv_fclss_d(rs1_data[t].u64);
			} else {
				rd_data[t].i = rv_fclss_s(check_boxing(rs1_data[t].u64));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMVXW: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rs1_data[t].u64;
			} else {
				uint32_t result = (uint32_t)rs1_data[t].u64;
				rd_data[t].i = sext((uint64_t)result, 32);
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMVWX: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rs1_data[t].i;
			} else {
				rd_data[t].u64 = nan_box((uint32_t)rs1_data[t].i);
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FNMADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fnmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FNMSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fnmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fnmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	default:
		std::abort();
	}
	DT(3, this->name() << " execute: op=" << fpu_type << ", " << *trace);
}

void FpuUnit::on_tick() {
	for (uint32_t b = 0; b < NUM_FPU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		if (output.full())
			continue; // stall
		auto trace = input.peek();
		this->execute(trace);
		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		input.pop();
	}
}
